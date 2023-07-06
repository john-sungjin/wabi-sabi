import os
import time

import datasets
import psutil
from composer.utils import reproducibility
from dotenv import load_dotenv
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = "/datadrive/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/datadrive/hf_cache"

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

###### CONFIG ######
tokenizer_save_dir = os.path.join(script_dir, "tokenizer/")
batch_size = 100
###### END CONFIG ######

if not os.path.exists(tokenizer_save_dir):
    os.makedirs(tokenizer_save_dir)


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024**2)} MB")


# Load all datasets
# streaming because https://huggingface.co/docs/datasets/v2.13.1/en/about_mapstyle_vs_iterable
seed = 42
reproducibility.seed_all(seed)

print("Loading dataset...")
start_time = time.time()
print_memory_usage()
tinystories_data: datasets.Dataset = datasets.load_from_disk(
    data_dir,
)  # type: ignore # created in create_dataset.py
print_memory_usage()
end_time = time.time()
print(f"Loading dataset took {end_time - start_time} seconds")

print("Dataset Sizes (in GB)")
print("Tiny Stories:", tinystories_data.info.splits["train"].num_bytes / (1024**3))  # type: ignore

# Creating tokenizer
dataset = tinystories_data
print("Dataset size:", dataset.num_rows)


def batch_generator(dataset: datasets.Dataset, batch_size: int):
    return (
        dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)
    )


tokenizer = Tokenizer(
    models.BPE(
        vocab=None,
        merges=None,
        unk_token=None,
        dropout=None,
        fuse_unk=False,
    )
)
tokenizer = Tokenizer(
    models.BPE(
        vocab=None,
        merges=None,
        unk_token=None,
        dropout=None,
        fuse_unk=False,
    )
)
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFKC(), normalizers.Lowercase(), normalizers.StripAccents()]
)  # type: ignore
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.ByteLevel(add_prefix_space=True),
        pre_tokenizers.Digits(individual_digits=True),
    ]
)  # type: ignore
tokenizer.post_processor = processors.ByteLevel()  # type: ignore
tokenizer.decoder = decoders.Sequence(
    [
        decoders.ByteLevel(),
    ]
)  # type: ignore
trainer = trainers.BpeTrainer(
    vocab_size=8192,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    # limit_alphabet=None,
    # initial_alphabet=None,
    show_progress=True,
)  # type: ignore

print("Training...")
start_time = time.time()
tokenizer.train_from_iterator(
    iterator=batch_generator(dataset, batch_size),
    trainer=trainer,
    length=dataset.num_rows,
)

end_time = time.time()
print(f"Training took {end_time - start_time} seconds")

print("Done training! Saving...")

tokenizer.save(os.path.join(tokenizer_save_dir, "tokenizer.json"))

pretrained_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    os.path.join(tokenizer_save_dir, "tokenizer.json")
)
pretrained_tokenizer.pad_token = "<|pad|>"
pretrained_tokenizer.pad_token_id = tokenizer.token_to_id("<|pad|>")

pretrained_tokenizer.save_pretrained(tokenizer_save_dir)
