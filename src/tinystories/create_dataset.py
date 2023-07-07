# Creating a Hugging Face Dataset from the text file.

# In[ ]:
# set working directory to root of project
import os

import datasets

CACHE_DIR = "/datadrive/hf_cache"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
raw_file = os.path.join(data_dir, "raw/TinyStoriesV2-GPT4-train.txt")


# In[ ]:
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


# In[ ]:
tinystories_data = datasets.Dataset.from_generator(
    tinystories_generator, cache_dir=CACHE_DIR
)
tinystories_data.save_to_disk(data_dir)

# In[ ]:
# Load the dataset
tinystories_data = datasets.load_from_disk(data_dir)


# In[ ]:
print(tinystories_data[1])


# In[ ]:
tinystories_instruct_data = datasets.load_dataset(
    "roneneldan/TinyStoriesInstruct", cache_dir=CACHE_DIR
)
