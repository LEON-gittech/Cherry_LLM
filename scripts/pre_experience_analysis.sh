python3 cherry_seletion/data_analysis.py \
    --data_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/data_concated.parquet \
    --save_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/pre_exp_data.pt \
    --model_name_or_path openai-community/gpt2 \
    --max_length 1024 \
    --prompt alpaca \
    --mod pre