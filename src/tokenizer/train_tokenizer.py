import os
import time

import datasets
import psutil
from composer.utils import reproducibility
from datasets import load_dataset
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


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024**2)} MB")


# Load all datasets
# streaming because https://huggingface.co/docs/datasets/v2.13.1/en/about_mapstyle_vs_iterable
seed = 42
reproducibility.seed_all(seed)

print("Loading datasets...")
print_memory_usage()
wikipedia_dataset: datasets.Dataset = load_dataset(
    "wikipedia",
    name="20220301.en",
    cache_dir=CACHE_DIR,
    use_auth_token=HF_TOKEN,
    split="train",
    # streaming=True,
).shuffle(
    seed=seed
)  # type: ignore
python_stack_dataset: datasets.Dataset = (
    load_dataset(
        "bigcode/the-stack-dedup",
        cache_dir=CACHE_DIR,
        data_dir="data/python",
        use_auth_token=HF_TOKEN,
        split="train",
        # streaming=True,
    )
    .shuffle(seed=seed)
    .rename_column("content", "text")
)  # type: ignore
wikihow_data: datasets.Dataset = load_dataset(
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
print_memory_usage()


print("Dataset Sizes (in GB)")
print("Wikipedia:", wikipedia_dataset.info.splits["train"].num_bytes / (1024**3))  # type: ignore
print("Python:", python_stack_dataset.info.splits["train"].num_bytes / (1024**3))  # type: ignore
print("Wikihow:", wikihow_data.info.splits["train"].num_bytes / (1024**3))  # type: ignore

# Creating tokenizer
# Want to control data mixture
# Wikipedia 30%,Python 40%, Wikihow 30%? Seems reasonable
print("Interleaving datasets...")
dataset = datasets.interleave_datasets(
    [wikipedia_dataset, python_stack_dataset, wikihow_data],
    probabilities=[0.3, 0.4, 0.3],
    seed=seed,
)
print("Done interleaving datasets.")
print("Dataset size:", dataset.num_rows)

# tokenizer = GPTNeoXTokenizerFast.from_pretrained(
#     "mosaicml/mpt-30b", cache_dir=CACHE_DIR
# )


def batch_generator(dataset: datasets.Dataset, batch_size: int):
    return (
        dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)
    )


# print("Training...")
# start_time = time.time()
# batch_size = 100
# new_tokenizer = tokenizer.train_new_from_iterator(
#     batch_generator(dataset, 100),
#     vocab_size=8192,
#     length=dataset.num_rows,
# )
# end_time = time.time()
# print(f"Training took {end_time - start_time} seconds")
# print("Done training! Saving...")
# new_tokenizer.save_pretrained("tokenizers/")


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
    special_tokens=["<|endoftext|>"],
    # limit_alphabet=None,
    # initial_alphabet=None,
    show_progress=True,
)  # type: ignore

print("Training...")
start_time = time.time()
tokenizer.train_from_iterator(
    iterator=batch_generator(dataset, 100), trainer=trainer, length=dataset.num_rows
)

end_time = time.time()
print(f"Training took {end_time - start_time} seconds")

print("Done training! Saving...")
tokenizer.save("tokenizer.json")

pretrained_tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer.json")
pretrained_tokenizer.pad_token = "<|endoftext|>"
pretrained_tokenizer.pad_token_id = tokenizer.token_to_id("<|endoftext|>")

pretrained_tokenizer.save_pretrained("tokenizer/")
