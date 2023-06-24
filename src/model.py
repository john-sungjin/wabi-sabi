import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from xformers.components.feedforward import FusedMLP
from xformers.triton import FusedLayerNorm


# need to fix this config
class WSConfig(PretrainedConfig):
    """
    Important ratios:
    - d_model should be a multiple of n_heads
    - d_q, d_k, d_v are all equal to d_model / n_heads
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        vocab_size: int = 50368,
        **kwargs,  # for other HuggingFace params
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_head = self.d_model // self.n_heads


class WSMultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()

        # NOTE: this layer concatenates several tensors together
        # When initializing params, we should instantiate these separately.
        # output size: q -> d_model, k and v -> head_dim
        self.residual_to_qkv = nn.Linear(d_model, d_model + 2 * d_head)
        # _splits[0]: the dimension by which the tensor is split
        # Note that, for nn.Linear(dim_a, dim_b), the weights are of shape (dim_b, dim_a)
        # So, the split dimension here is 0
        # _splits[1]: the split sizes
        self.residual_to_qkv._splits = (0, (d_model, d_head, d_head))

        self.concat_attention_to_residual = nn.Linear(d_model, d_model)
        self.concat_attention_to_residual._is_residual_projection = True

    def forward(self, x: torch.Tensor):
        qkv = self.residual_to_qkv(x)
        # split qkv into q, k, v
        q, k, v = torch.split(qkv, [self.d_model, self.d_head, self.d_head], dim=-1)

        # PyTorch's flash attention implementation
        concat_attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.concat_attention_to_residual(concat_attention)
        return x


# Flash attention, multi-query attention
class WSBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
    ):
        super().__init__()
        self.layer_norm_before_attention = FusedLayerNorm(d_model)
        self.attention = WSMultiQueryAttention(d_model, d_head)
        self.layer_norm_before_ffn = FusedLayerNorm(d_model)

        # this ratio is standard for transformers
        # Triton fused MLP. The linear layers in PyTorch are already fused,
        # but xformers has a custom implementation that fuses dropout and bias
        # not useful since there is no bias... but I'll use it anyway
        ffn_ratio = 4
        self.ffn = FusedMLP(
            dim_model=d_model,
            dropout=0.0,
            activation="gelu",
            hidden_layer_multiplier=ffn_ratio,
        )
        # index of the last linear layer
        self.ffn.mlp[2]._is_residual_projection = True

    def forward(self, x: torch.Tensor):
        x_norm = self.layer_norm_before_attention(x)
        x_attn = self.attention(x_norm)
        x = x + x_attn

        x_norm = self.layer_norm_before_ffn(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn
        return x


class WSModel(PreTrainedModel):
    def __init__(self, config: WSConfig):
        super().__init__(config)
        self.config = config
        self.tokens_to_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model
        )

        # Embedding fraction: page 7 of GLM-130B paper https://arxiv.org/abs/2210.02414
        # self.embedding_fraction = config.embedding_fraction

        self.blocks = nn.ModuleList(
            [
                WSBlock(
                    d_model=config.d_model,
                    d_head=config.d_head,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Could also consider the Mosaic implementation...
        # https://docs.mosaicml.com/projects/composer/en/latest/method_cards/low_precision_layernorm.html
        self.layer_norm_final = FusedLayerNorm(config.d_model)

        self.embeddings_to_logits = nn.Linear(
            in_features=config.d_model, out_features=config.vocab_size
        )

        # https://paperswithcode.com/method/weight-tying
        self.embeddings_to_logits.weight = self.tokens_to_embeddings.weight

        # initialize parameters
        # notes from MPT/nanoGPT/transformers
        # 1. residual projections (e.g. linear layers that project to d_model) are divided
        # by sqrt(num_layers)
        # 2. layer norm weights are set to one (PyTorch sets this by default; skip)
        # 3. all others are initialized with normal distribution with mean 0 and std 0.02
        # Note: MPT uses kaiming_normal; I'll go for this as well
        def init_weights(module: nn.Module):
            if isinstance(module, nn.Linear):
                if hasattr(module, "_splits"):
                    split_dim, split_sizes = module._splits
                    assert module.weight.shape[split_dim] == sum(split_sizes)
                    start = 0
                    for size in split_sizes:
                        slice_indices = [slice(None)] * module.weight.ndim
                        slice_indices[split_dim] = slice(start, start + size)
                        nn.init.kaiming_normal_(module.weight[slice_indices])
                        start += size
                    return

                nn.init.kaiming_normal_(module.weight)
                if getattr(module, "_is_residual_projection", False):
                    with torch.no_grad():
                        module.weight.div_(math.sqrt(config.n_layers))

            elif isinstance(module, nn.Embedding):
                nn.init.kaiming_normal_(module.weight)

        # disable bias in all modules
        # note for later: if you want to enable bias, should remember to zero out all biases
        # in init_weights
        def disable_bias(module: nn.Module):
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                module.register_parameter("bias", None)

        self.apply(init_weights)
        self.apply(disable_bias)

        # print model stats
        # num parameters, num flops, num bytes

    # TODO: kwargs are for other HuggingFace generate params. Implement if needed.
    def forward(self, input_ids: torch.LongTensor, **kwargs):
        x = self.tokens_to_embeddings(input_ids)

        # MPT doesn't use embedding fraction
        # x = (x * self.embedding_fraction) + (x.detach() * (1 - self.embedding_fraction))

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm_final(x)
        x = self.embeddings_to_logits(x)
        return x
