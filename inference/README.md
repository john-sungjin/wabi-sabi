To get weights:
```
python llm-foundry/scripts/inference/convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o mpt-ft-7b --infer_gpu_num 1
python llm-foundry/scripts/inference/convert_hf_mpt_to_ft.py -i mosaicml/mpt-30b -o mpt-ft-30b --infer_gpu_num 1 --weight_data_type fp16
```

Command to run from this directory, in the FT container:
`PYTHONPATH=FasterTransformer FT_LOG_LEVEL=DEBUG python llm-foundry/scripts/inference/run_mpt_with_ft.py --ckpt_path mpt-ft-7b/1-gpu/ --lib_path FasterTransformer/build/lib/libth_transformer.so --sample_input_file prompts.txt --sample_output_file output.txt`
`PYTHONPATH=FasterTransformer FT_LOG_LEVEL=DEBUG python llm-foundry/scripts/inference/run_mpt_with_ft.py --ckpt_path mpt-ft-30b/1-gpu/ --lib_path FasterTransformer/build/lib/libth_transformer.so --sample_input_file prompts.txt --sample_output_file output.txt`