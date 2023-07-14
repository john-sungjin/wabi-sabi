# In[]
import json
import os
import random
import time
from typing import Any

import datasets
import numpy as np
import torch
import transformers

HF_CACHE = "/datadrive/hf_cache"
data_dir = "/datadrive/wabi-sabi/data/tinystories"
seed = 42

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

batch_size = 64
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
name = "mosaicml/mpt-30b"
config = transformers.AutoConfig.from_pretrained(
    name, trust_remote_code=True, cache_dir=HF_CACHE
)
config.attn_config["attn_impl"] = "triton"
config.init_device = device
config.max_seq_len = max_seq_len

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir=HF_CACHE,
)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(name, cache_dir=HF_CACHE)
tokenizer.pad_token = tokenizer.eos_token  # doesn't have a pad token

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
        max_length=config.max_seq_len,
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

train_dataloader = torch.utils.data.DataLoader(
    tokenized_data,
    batch_size=batch_size,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=16,
    persistent_workers=True,
)

# In[]

n_active_params = sum(p.numel() for p in model.parameters())


def flops_per_batch(batch, train=False):
    """
    Estimates the number of flops per batch.
    """
    bs, msl = batch["input_ids"].shape[0:2]
    params_flops_per_token = 2 * n_active_params
    params_flops_per_seq = params_flops_per_token * msl
    attn_flops_per_seq = (
        model.config.n_layers * 2 * 2 * (model.config.d_model * (msl**2))
    )

    multiplier = 1
    if train:
        # backward pass is 2x the forward pass
        multiplier = 3

    return (params_flops_per_seq + attn_flops_per_seq) * multiplier * bs


def write_outputs():
    """
    Passes in batches into the model to generate logits.
    Not using Dataset.from_generator because it tries to pickle the entire generator...?
    """
    with open(os.path.join(data_dir, "logits"), "w") as outfile:
        with torch.inference_mode():
            counter = 0
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(device)

                attention_mask = batch["attention_mask"].to(device)

                # measure forward pass time
                torch.cuda.synchronize()
                start = time.time()
                output = model(input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()
                end = time.time()

                time_elapsed_s = end - start
                # print(end="\x1b[2K")
                print(
                    "Time: {:.3f}s, TFLOPS: {:.3f}".format(
                        time_elapsed_s, flops_per_batch(batch) / (time_elapsed_s * 1e12)
                    ),
                    # end="\r",
                )

                input_ids = input_ids.to("cpu", dtype=torch.float16)
                attention_mask = attention_mask.to("cpu", dtype=torch.float16)
                logits = output.logits.to("cpu", dtype=torch.float16)

                for i in range(input_ids.shape[0]):
                    # get subset of input_ids where attention_mask == 1
                    sample_input_ids = input_ids[i][attention_mask[i] == 1].tolist()
                    sample_logits = logits[i][attention_mask[i] == 1].tolist()

                    outfile.write(
                        json.dumps(
                            {
                                "input_ids": sample_input_ids,
                                "logits": sample_logits,
                            }
                        )
                    )

                    print("Written!")

                    break

                break

                # counter += 1
                # if counter > 100:
                #     break


write_outputs()


# %%
