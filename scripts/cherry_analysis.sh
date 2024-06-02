python3 cherry_seletion/data_analysis.py \
    --data_path data/alpaca_data.json \
    --save_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_cherry.pt \
    --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/pre-alpaca/ \
    --max_length 2048 \
    --prompt alpaca \
    --mod cherry

# torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 cherry_seletion/data_analysis.py     --data_path data/alpaca_data.json     --save_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_cherry.pt     --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/pre-alpaca/     --max_length 2048     --prompt alpaca     --mod cherry