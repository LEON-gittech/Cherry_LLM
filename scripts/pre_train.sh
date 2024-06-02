CUDA_VISIBLE_DEVICES=2 python3 /opt/tiger/Cherry_LLM/training/stanford_alpaca/train.py \
  --model_name_or_path "/mnt/bn/data-tns-live-llm/leon/datasets/llama-2-7b-bnb-4bit"
  --data_path /mnt/bn/data-tns-live-llm/leon/datasets/repetition/alpaca_data_pre.json \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --eval_strategy no \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --output_dir /mnt/bn/data-tns-live-llm/leon/datasets/repetition/pre-alpaca