# In[ ]:
from transformers import PreTrainedTokenizerFast

tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
    "tinystories/tokenizer"
)

prompt = "Once upon a time, in five"
print(tokenizer.encode(prompt))
