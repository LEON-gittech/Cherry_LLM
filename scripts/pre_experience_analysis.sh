python3 cherry_seletion/data_analysis.py \
    --data_path data/alpaca_data.json \
    --save_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_pre.pt \
    --model_name_or_path yahma/llama-7b-hf \
    --max_length 512 \
    --prompt alpaca \
    --mod pre