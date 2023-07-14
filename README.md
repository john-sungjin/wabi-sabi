# wabi-sabi
A wabi-sabi LLM

This git repo holds a bunch of work that I've been doing to increase my practical knowledge of working with language models. A few things I've done:
- Trained models from scratch, including the tokenizer (Hugging Face and MosaicML Composer)
- Implemented my model in GGML


TODO:
- Inference/deployment: hook my model up to a framework and deploy it, accessible through API
- Fine-Tuning (LORA, SFT)
- Quantization
- Large models: how to train/deploy on multiple GPUs, getting more intuition here

To run docker container for inference:
```
docker run -it --gpus all -v /datadrive:/datadrive --name main --shm-size=8g mosaicml/llm-foundr
y:2.0.1_cu118-latest /bin/bash
```