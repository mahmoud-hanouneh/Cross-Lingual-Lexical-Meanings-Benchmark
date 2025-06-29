lm_eval `    --model hf`
--model_args pretrained=microsoft/Phi-3-mini-4k-instruct,dtype=float16,trust_remote_code=True `    --tasks csi_custom_task`
--include_path tasks `    --batch_size 4`
--output_path ./results/harness_phi3_results.json
