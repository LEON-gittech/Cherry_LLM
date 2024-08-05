python3 cherry_seletion/data_by_cluster.py \
    --pt_data_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/pre_exp_data.pt \
    --json_data_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/data_concated.parquet \
    --json_save_path /mnt/bn/data-tns-live-llm/leon/datasets/fed_data/pre_exp_selection_data.json \
    --sample_num 10 \
    --kmeans_num_clusters 100 \
    --low_th 25 \
    --up_th 75