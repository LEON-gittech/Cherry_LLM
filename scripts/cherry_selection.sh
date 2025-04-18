python3 cherry_seletion/data_by_IFD.py \
    --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/pre-alpaca/ \
    --pt_data_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_cherry.pt \
    --json_data_path data/alpaca_data.json \
    --json_save_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_cherry_5per.json \
    --max_length 2048 \
    --sample_rate 0.06 \
    --prompt alpaca

python3 cherry_seletion/quality_analyse.py \
    --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/gpt2_pre_exp/ \
    --pt_data_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/cherry_iid2niid_med.pt \
    --json_data_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/iid2niid_med_concated.parquet \
    --json_save_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_cherry_5per.json \
    --max_length 1024 \
    --sample_rate 0.06 \
    --prompt alpaca