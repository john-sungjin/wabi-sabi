import os
from typing import Any

import datasets
import torch.utils.data
import wandb
from composer import Callback, Logger, State, Time, Trainer
from composer.callbacks import SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from composer.utils import reproducibility
from model import ComposerWSModel, WSConfig
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = "/datadrive/hf_cache"

###### CONFIG ######
model_params = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "vocab_size": 8192,
}

seed = 42
optim = {
    "lr": 1e-4,
    "betas": (0.9, 0.98),
    "eps": 1.0e-06,
    "weight_decay": 1.0e-5,
}
learning_rate = {"t_warmup": "250ba", "alpha_f": 0.02}
precision = "fp32"
batch_size = 64
context_length = 256

save_folder = "checkpoints/pretraining/"
save_interval = "500ba"
hf_save_folder = "huggingface_model/"

tokenizer_dir = "tokenizer/"
###### END CONFIG ######


reproducibility.seed_all(seed)

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
config = WSConfig(**model_params)

text_column_name = "text"


def tokenize_function(examples: dict[str, Any]):
    """
    Tokenize dataset examples.
    """
    examples[text_column_name] = [
        line
        for line in examples[text_column_name]
        if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples[text_column_name],
        padding="max_length",
        truncation=True,
        max_length=context_length,
        return_special_tokens_mask=True,
    )


print("Loading datasets...")
wikihow_data: datasets.Dataset = datasets.load_dataset(
    "wikihow",
    name="all",
    data_dir=CACHE_DIR,
    cache_dir=CACHE_DIR,
    use_auth_token=HF_TOKEN,
    split="train",
    # streaming=True,
).shuffle(
    seed=seed
)  # type: ignore

tokenized_train = wikihow_data.map(
    tokenize_function,
    batched=True,
    remove_columns=wikihow_data.column_names,  # collate_fn doesn't like other columns
    load_from_cache_file=False,
)

collate_fn = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

train_dataloader = torch.utils.data.DataLoader(
    tokenized_train, batch_size=batch_size, collate_fn=collate_fn
)

composer_model = ComposerWSModel(config=config, tokenizer=tokenizer)
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
            max_new_tokens=30,
        )
        output_text = self.tokenizer.decode(output_ids[0])
        self.table.add_data(output_text)
        logger.log_metrics({"samples": self.table})

        self.last_sample = state.timestamp.batch


wandb_logger = WandBLogger(project="wabisabi")

# Create Trainer Object
trainer = Trainer(
    model=composer_model,  # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    # eval_dataloader=eval_dataloader,
    max_duration="1ep",  # train for more epochs to get better performance
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    device="gpu" if torch.cuda.is_available() else "cpu",
    precision="fp32",
    progress_bar=True,
    loggers=[wandb_logger],
    callbacks=[
        SpeedMonitor(),
        SampleCallback("Hi, my name is", tokenizer, save_interval),
    ],
    # checkpointing
    save_folder=save_folder,
    save_filename="ep{epoch}-ba{batch}-rank{rank}.pt",
    save_interval=save_interval,
    save_overwrite=True,
)

# Start training
trainer.fit()

# Save Hugging Face model
config.save_pretrained(hf_save_folder)
tokenizer.save_pretrained(hf_save_folder)
composer_model.model.save_pretrained(hf_save_folder)
