# array=(
#     vicuna
#     koala
#     wizardlm
#     sinstruct
#     lima
# )
# for i in "${array[@]}"
# do
#     echo $i
#         python evaluation/generation/eva_generation.py \
#             --dataset_name $i \
#             --model_name_or_path xxx \
#             --max_length 1024 

# done

CUDA_VISIBLE_DEVICES=2 python3 evaluation/generation/eva_generation.py \
            --dataset_name vicuna \
            --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/cherry-code/ \
            --max_length 2048

CUDA_VISIBLE_DEVICES=0 python3 evaluation/generation/eva_generation.py \
            --dataset_name vicuna \
            --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/MoDS-code/ \
            --max_length 2048