To get the Composer GPU dependencies:
```
pip install flash-attn --no-build-isolation
pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git#subdirectory=cs
rc/xentropy --no-build-isolation
```

These don't install correctly for some reason with Poetry.

Just kidding, just use the Mosaic ML Docker container.

Also, I could've realized in hindsight, but this is . . . way too much data.

Each batch of 64 samples with 512 tokens, for a vocab size of ~50000 and fp16 logits comes out to about 3.2 GB. This is going to get impractical really quickly...

The new goal: collocated models.

Create some kind of a harness that uses a large model to generate logits/text, and a smaller model to train off those logits/that text.

```
docker run -it --gpus all -v /datadrive:/datadrive --name main --shm-size=8g mosaicml/llm-foundr
y:2.0.1_cu118-latest /bin/bash
```

Conversion
```
python ../../../../inference/llm-foundry/scripts/inference/convert_composer_to_hf.py --composer_path checkpoints/ep0-ba1000-rank0.pt --hf_output_path hf_model/ --output_precision bf16
```

Findings: model doesn't learn from this script, prefers to just repeat "in"
Might read some papers, also feel like I should normalize the logits or something