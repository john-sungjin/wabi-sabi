from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer.json")
test_str = "Hello, world! 12345"
encoded = tokenizer(test_str)
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded["input_ids"])
print("Decoded:", decoded)
