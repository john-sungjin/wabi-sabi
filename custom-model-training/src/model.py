import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics.nlp import LanguageCrossEntropy
from composer.models.huggingface import HuggingFaceModel
from einops import rearrange, repeat
from rotary_embedding import RotaryEmbedding
from torchmetrics import Metric
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# from xformers.components.feedforward import FusedMLP
# from xformers.triton import FusedLayerNorm


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
        n_heads: int = 4,
        n_layers: int = 2,
        vocab_size: int = 8192,
        **kwargs: Any,  # for other HuggingFace params
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_head = self.d_model // self.n_heads


class WSMultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
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

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x: torch.Tensor):
        qkv = self.residual_to_qkv(x)
        # split qkv into q, k, v
        q, k, v = torch.split(qkv, [self.d_model, self.d_head, self.d_head], dim=-1)

        # PyTorch's flash attention implementation
        # the function expects q and k to have the same last dimension
        # what we'll do is go from (batch_size, seq_len, d_model) to
        # (batch_size, seq_len, n_heads, d_head) for q, k, and v
        # q has values for all n_heads; since this is multiquery,
        # k and v will have n_heads = 1, then get expanded to n_heads
        # this is what mosaic does

        q = rearrange(
            q,
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )
        # verified that this does the same as torch.expand; sets stride to 0
        k = repeat(
            k,
            "batch seq_len (1 d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )
        v = repeat(
            v,
            "batch seq_len (1 d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )

        concat_attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # get back to (batch_size, seq_len, d_model)
        concat_attention = rearrange(
            concat_attention,
            "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )
        x = self.concat_attention_to_residual(concat_attention)

        return x


class WSAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()

        # NOTE: this layer concatenates several tensors together
        # When initializing params, we should instantiate these separately.
        # output size: q, k, and v are all -> d_model
        self.residual_to_qkv = nn.Linear(d_model, d_model * 3)
        # _splits[0]: the dimension by which the tensor is split
        # Note that, for nn.Linear(dim_a, dim_b), the weights are of shape (dim_b, dim_a)
        # So, the split dimension here is 0
        # _splits[1]: the split sizes
        self.residual_to_qkv._splits = (0, (d_model, d_model, d_model))

        self.concat_attention_to_residual = nn.Linear(d_model, d_model)
        self.concat_attention_to_residual._is_residual_projection = True

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        # no learned parameters, so technically can be instantiated once
        # clean up later
        # self.rotary_embedding = RotaryEmbedding(d_head)

    def forward(self, x: torch.Tensor):
        qkv = self.residual_to_qkv(x)
        # split qkv into q, k, v
        q, k, v = torch.split(qkv, [self.d_model, self.d_model, self.d_model], dim=-1)

        # PyTorch's flash attention implementation
        # the function expects q and k to have the same last dimension
        # what we'll do is go from (batch_size, seq_len, d_model) to
        # (batch_size, seq_len, n_heads, d_head) for q, k, and v
        # q has values for all n_heads; since this is multiquery,
        # k and v will have n_heads = 1, then get expanded to n_heads
        # this is what mosaic does

        q = rearrange(
            q,
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )
        # verified that this does the same as torch.expand; sets stride to 0
        k = rearrange(
            k,
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )

        # apply rotary embedding to q and k
        # q = self.rotary_embedding.rotate_queries_or_keys(q)
        # k = self.rotary_embedding.rotate_queries_or_keys(k)

        v = rearrange(
            v,
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )

        concat_attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # get back to (batch_size, seq_len, d_model)
        concat_attention = rearrange(
            concat_attention,
            "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)",
            n_heads=self.n_heads,
            d_head=self.d_head,
        )
        x = self.concat_attention_to_residual(concat_attention)

        return x


# Flash attention, multi-query attention
class WSBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
    ):
        super().__init__()
        self.layer_norm_before_attention = FusedLayerNorm(d_model)
        # self.attention = WSMultiQueryAttention(d_model, n_heads, d_head)
        self.attention = WSAttention(d_model, n_heads, d_head)
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
    """
    Generally, Hugging Face LLMs have a base model that outputs hidden states,
    then a head that outputs the task-specific outputs. I'm only implementing
    autoregressive language modeling, so I'll just have it all here for simplicity.
    """

    # Need to define this for Hugging Face
    # "The line that sets the config_class is not mandatory, unless you want to register your model with the auto classes (see last section)."
    # config_class = WSConfig

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
                    n_heads=config.n_heads,
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        **kwargs: Any,
    ):
        """
        Needs to return a CausalLMOutputWithPast. In the generation loop, we expect
        output to have attrs logits, decoder_attentions (if output_attentions),
        and hidden_states (if output_hidden_states).
        """
        # input_ids = input_ids.long() # have this for deepspeed profiling
        x = self.tokens_to_embeddings(input_ids)

        # MPT doesn't use embedding fraction
        # x = (x * self.embedding_fraction) + (x.detach() * (1 - self.embedding_fraction))

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm_final(x)
        x = self.embeddings_to_logits(x)
        return CausalLMOutputWithPast(logits=x)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs: Any):
        """
        Must be implemented to use generate() method. Checked by can_generate().

        Not sure what the other kwargs are for, but you can see what the function expects
        in GenerationMixin.

        Expects a dictionary model_inputs. This is destructured and given to the
        forward() method.
        """

        return {"input_ids": input_ids}


class ComposerWSModel(HuggingFaceModel):
    def __init__(
        self,
        config: WSConfig,
        tokenizer: PreTrainedTokenizerFast,
    ):
        model = WSModel(config)

        # this takes in pred and target logits
        # should be batch_size x seq_len x vocab_size? probably
        train_metrics: list[Metric] = [LanguageCrossEntropy()]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            shift_labels=True,  # labels come in unshifted, this shifts before metrics
            metrics=train_metrics,
        )

        # Note: wanted to use flash-attn for fused CE, but there's an install error with rye
        # Honestly should be pretty small relative to other things, not going to worry about it for now
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Used to compute flops
        self.n_active_params = sum(p.numel() for p in self.parameters())

    def forward(self, batch: dict[str, Any]):
        """
        Mosaic's forward pass. Batch is a Mapping with keys possibly reflecting HuggingFace's forward function inputs.
        Check GPT2 implementation for args; there isn't really a standard set.

        Output needs to be an output dataclass from huggingface.
        """
        return self.model(batch["input_ids"])

    def loss(self, outputs: CausalLMOutputWithPast, batch: dict[str, Any]):
        """
        Mosaic's loss function. Outputs is the output of the forward pass.
        """
        # outputs is batch x seq_len x vocab_size
        # labels is batch x seq_len
        # need to reduce to (batch * seq_len) x vocab_size and (batch * seq_len)
        output_logits = rearrange(
            outputs.logits,
            "batch seq_len vocab_size -> (batch seq_len) vocab_size",
            vocab_size=self.config.vocab_size,
        )
        labels = batch["labels"]
        # shift labels left
        labels = torch.roll(labels, -1, dims=1)
        labels[:, -1] = -100  # don't predict the last token
        # flatten
        labels = rearrange(labels, "batch seq_len -> (batch seq_len)")

        return self.loss_fn(output_logits, labels)

    def flops_per_batch(self, batch: dict[str, Any]):
        """
        Taken from MPT. MSL is max sequence length.
        """
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch["input_ids"].shape[0:2]
        params_flops_per_token = 2 * self.n_active_params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (
            self.model.config.n_layers
            * 2
            * 2
            * (self.model.config.d_model * (msl**2))
        )

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
