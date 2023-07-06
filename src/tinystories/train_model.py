# In[ ]:
import os
import sys
from typing import Any

import datasets
import torch.utils.data
import torchinfo
import wandb
from composer import Callback, Logger, State, Time, Trainer
from composer.callbacks import SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from composer.utils import reproducibility
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

# to allow model import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import ComposerWSModel, WSConfig

HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = "/datadrive/hf_cache"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# In[ ]:
###### CONFIG ######
run_name = "ts-1"  # change if you want to not autoresume

run_dir = os.path.join(script_dir, "runs", run_name)

model_params = {
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 12,
    "vocab_size": 8192,
}

seed = 42
optim = {
    "lr": 6.0e-4,
    "betas": (0.9, 0.95),
    "eps": 1.0e-08,
    "weight_decay": 0.0,
}
learning_rate = {"t_warmup": "100ba", "alpha_f": 0.1}
precision = "fp32"
batch_size = 128
context_length = 256  # removing this because we want to train on full stories

save_folder = os.path.join(run_dir, "checkpoints")
save_interval = "500ba"
hf_save_folder = os.path.join(run_dir, "hfmodel")

tokenizer_dir = os.path.join(script_dir, "tokenizer")
###### END CONFIG ######
reproducibility.seed_all(seed)

# In[ ]:
tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
    tokenizer_dir
)
config = WSConfig(**model_params)

text_column_name = "text"


def tokenize_function(examples: dict[str, Any]):
    """
    Tokenize dataset examples.
    We don't truncate anything for Tiny Stories;
    we'll do max length padding in the collate function.
    """
    examples[text_column_name] = [
        line
        for line in examples[text_column_name]
        if len(line) > 0 and not line.isspace()
    ]

    tokenized = tokenizer(
        examples[text_column_name],
        return_special_tokens_mask=True,
        padding="max_length",
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
    )

    return tokenized


print("Loading datasets...")
tinystories_data: datasets.Dataset = datasets.load_from_disk(data_dir).with_format("torch")  # type: ignore

# tinystories_data = tinystories_data.select(range(100))

tokenized_train = tinystories_data.map(
    tokenize_function,
    batched=True,
    remove_columns=tinystories_data.column_names,  # collate_fn doesn't like other columns
    num_proc=8,
)

print("Length of dataset before filtering:", tokenized_train.num_rows)
# tokenized_train = tokenized_train.filter(
#     lambda x: len(x["input_ids"]) <= context_length, num_proc=8
# )
# print("Length of dataset after filtering:", tokenized_train.num_rows)

# In[ ]:
# this collator sets padding tokens to -100
# before, we had set padding tokens = endoftext, so model wasn't learning
# to predict the end of the story
collate_fn = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

train_dataloader = torch.utils.data.DataLoader(
    tokenized_train,
    batch_size=batch_size,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
)

# In[ ]:
composer_model = ComposerWSModel(config=config, tokenizer=tokenizer)
print(
    torchinfo.summary(
        composer_model.model,
        input_size=(batch_size, context_length),
        dtypes=[torch.long],
    )
)

optimizer = DecoupledAdamW(
    composer_model.model.parameters(),
    **optim,
)
lr_scheduler = LinearWithWarmupScheduler(**learning_rate)


class SampleCallback(Callback):
    def __init__(
        self, sample_prompt: str, tokenizer: PreTrainedTokenizerFast, interval: str
    ):
        self.sample_prompt_ids = tokenizer.encode(sample_prompt, return_tensors="pt")
        self.interval = Time.from_timestring(interval)
        self.last_sample = Time(0, "ba")
        self.tokenizer = tokenizer

        # create table for samples
        self.table = wandb.Table(columns=["sample"])
        super().__init__()

    def batch_end(self, state: State, logger: Logger):
        if (state.timestamp.batch - self.last_sample) < self.interval:
            return
        output_ids = state.model.generate(
            state.device.tensor_to_device(self.sample_prompt_ids),
            max_new_tokens=100,
        )
        output_text = self.tokenizer.decode(output_ids[0])
        self.table.add_data(output_text)
        logger.log_metrics({"samples": self.table})

        self.last_sample = state.timestamp.batch


# In[ ]:
# Create Trainer Object
wandb_logger = WandBLogger(project="wabisabi")
trainer = Trainer(
    model=composer_model,  # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    # eval_dataloader=eval_dataloader,
    max_duration="1ep",  # train for more epochs to get better performance
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    device="gpu",
    precision="amp_fp16",  # mixed precision training
    progress_bar=True,
    loggers=[wandb_logger],
    callbacks=[
        SpeedMonitor(),
        SampleCallback("Once upon a time,", tokenizer, save_interval),
    ],
    # checkpointing
    save_folder=save_folder,
    save_filename="ep{epoch}-ba{batch}-rank{rank}.pt",
    save_interval=save_interval,
    # save_overwrite=True,
    # autoresume
    run_name=run_name,
    autoresume=True,
)

# Start training
trainer.fit()

# In[ ]:
print("Saving model...")
# Save Hugging Face model
config.save_pretrained(hf_save_folder)
tokenizer.save_pretrained(hf_save_folder)
composer_model.model.save_pretrained(hf_save_folder)
print("Done!")

trainer.close()
