# In[]
import os
import random
import time
from typing import Any

import composer
import composer.models
import datasets
import llmfoundry
import numpy as np
import omegaconf
import torch
import torch.utils.data
import transformers

HF_CACHE = "/datadrive/hf_cache"
data_dir = "/datadrive/wabi-sabi/data/tinystories"
root_dir = "/datadrive/wabi-sabi/distillation/src/"
student_config = os.path.join(root_dir, "student.yaml")
seed = 42

run_name = "test1"
run_dir = os.path.join(root_dir, "runs", run_name)
save_folder = os.path.join(run_dir, "checkpoints")
max_duration = "1000ba"
save_interval = "200ba"

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

batch_size = 8
max_seq_len = 512
device = "cuda:0"
generate_data = False

# In[]
# Hard coding in TinyStories for this script
# Prereq: wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
if generate_data:
    raw_file = os.path.join(data_dir, "raw/TinyStoriesV2-GPT4-train.txt")

    def tinystories_generator():
        """
        Generator for Tiny Stories.
        Stories in the text file are separated by <|endoftext|> tokens. We want to
        generate each complete story, ending with the <|endoftext|> token.
        """

        current_story = ""
        with open(raw_file, "r") as f:
            for line in f:
                if line.startswith("<|endoftext|>"):
                    yield {"text": current_story + "<|endoftext|>"}
                    current_story = ""
                else:
                    current_story += line

    tinystories_data = datasets.Dataset.from_generator(
        tinystories_generator, cache_dir=HF_CACHE
    )
    tinystories_data.save_to_disk(data_dir)

# In[]
teacher_name = "mosaicml/mpt-30b"
tokenizer = transformers.AutoTokenizer.from_pretrained(teacher_name, cache_dir=HF_CACHE)
tokenizer.pad_token = tokenizer.eos_token  # doesn't have a pad token

# In[]
# Teacher model initialization
# TODO: move to llm-foundry? with yaml config
teacher_config = transformers.AutoConfig.from_pretrained(
    teacher_name, trust_remote_code=True, cache_dir=HF_CACHE
)
teacher_config.attn_config["attn_impl"] = "triton"
teacher_config.init_device = device
teacher_config.max_seq_len = max_seq_len

teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
    teacher_name,
    config=teacher_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir=HF_CACHE,
)
teacher_model.n_active_params = sum(p.numel() for p in teacher_model.parameters())
teacher_model.eval()
# Freeze all the layers
# we do this here because we can't use the inference_mode context without
# preventing training everywhere
for param in teacher_model.parameters():
    param.requires_grad = False

# In[]
# Student model initialization
with open(student_config, "r") as f:
    yaml_cfg = omegaconf.OmegaConf.load(f)


class StudentModel(composer.models.HuggingFaceModel):
    def __init__(
        self,
        om_model_config,
        tokenizer,
    ):
        resolved_om_model_config = omegaconf.OmegaConf.to_container(
            om_model_config, resolve=True
        )
        hf_config = llmfoundry.MPTConfig.from_dict(resolved_om_model_config)
        model = llmfoundry.MPTForCausalLM(hf_config)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=False,
            shift_labels=False,
        )

        self.n_active_params = sum(p.numel() for p in self.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, batch):
        return self.model(
            input_ids=torch.squeeze(batch["input_ids"]),  # squeeze because batch size 1
            attention_mask=torch.squeeze(batch["attention_mask"]),
        )

    def loss(self, outputs, batch):
        mask = torch.permute(batch["attention_mask"], (1, 2, 0)) == 1
        logits = outputs.logits.masked_select(mask)
        targets = torch.squeeze(batch["logits"]).masked_select(mask)

        return self.loss_fn(targets, logits)


student_model = StudentModel(om_model_config=yaml_cfg.model, tokenizer=tokenizer)

# In[]
tinystories_data = datasets.load_from_disk(data_dir).with_format("torch")


def tokenize_function(examples: dict[str, Any]):
    """
    Tokenize dataset examples.
    We don't truncate anything for Tiny Stories;
    we'll do max length padding in the collate function.
    """
    text_column_name = "text"

    examples[text_column_name] = [
        line
        for line in examples[text_column_name]
        if len(line) > 0 and not line.isspace()
    ]

    tokenized = tokenizer(
        examples[text_column_name],
        return_special_tokens_mask=True,
        padding=False,
        max_length=teacher_config.max_seq_len,
        truncation=True,
    )

    return tokenized


tokenized_data = tinystories_data.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=tinystories_data.column_names,  # collate_fn doesn't like other columns
)

collate_fn = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

teacher_dataloader = torch.utils.data.DataLoader(
    tokenized_data,
    batch_size=batch_size,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=16,
    persistent_workers=True,
)

# In[]


def flops_per_batch(model, batch, train=False):
    """
    Estimates the number of flops per batch.
    """
    bs, msl = batch["input_ids"].shape[0:2]
    params_flops_per_token = 2 * model.n_active_params
    params_flops_per_seq = params_flops_per_token * msl
    attn_flops_per_seq = (
        model.config.n_layers * 2 * 2 * (model.config.d_model * (msl**2))
    )

    multiplier = 1
    if train:
        # backward pass is 2x the forward pass
        multiplier = 3

    return (params_flops_per_seq + attn_flops_per_seq) * multiplier * bs


def student_generator():
    # with torch.inference_mode():
    for batch in teacher_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # measure forward pass time
        torch.cuda.synchronize()
        start = time.time()
        output = teacher_model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        end = time.time()

        time_elapsed_s = end - start
        # print(end="\x1b[2K")
        print(
            "Time: {:.3f}s, TFLOPS: {:.3f}".format(
                time_elapsed_s,
                flops_per_batch(teacher_model, batch) / (time_elapsed_s * 1e12),
            ),
            # end="\r",
        )

        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logits": output.logits,
        }


class StudentDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


student_dataset = StudentDataset(student_generator)
student_loader = torch.utils.data.DataLoader(student_dataset, batch_size=1)


# %%
optimizer = llmfoundry.utils.builders.build_optimizer(yaml_cfg.optimizer, student_model)
lr_scheduler = llmfoundry.utils.builders.build_scheduler(yaml_cfg.scheduler)

wandb_logger = composer.loggers.WandBLogger(project="distillation")
trainer = composer.Trainer(
    model=student_model,  # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=student_loader,
    max_duration=max_duration,  # train for more epochs to get better performance
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    device="gpu",
    precision="amp_bf16",  # mixed precision training
    progress_bar=True,
    loggers=[wandb_logger],
    callbacks=[
        composer.callbacks.SpeedMonitor(),
    ],
    # checkpointing
    save_folder=save_folder,
    save_filename="ep{epoch}-ba{batch}-rank{rank}.pt",
    save_interval=save_interval,
    # save_overwrite=True,
    # autoresume
    run_name=run_name,
    # autoresume=True,
)

# Start training
trainer.fit()

# %%
