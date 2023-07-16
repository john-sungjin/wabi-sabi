# %%
import transformers

model_path = "/datadrive/wabi-sabi/distillation/src/runs/test7_no_teacher/hf_model"

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # doesn't have a pad token

# %%
prompt = "There was a boy named Jack. He was a good boy, and he was very good at sports. One day,"
prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

print("Prompt IDs:", prompt_ids)

output_ids = model.generate(prompt_ids, max_new_tokens=200)
print("Output IDs:", output_ids)

output_text = tokenizer.decode(output_ids[0])
print("Output Text:", output_text)

# %%
