#!/bin/bash

python fine-tune.py \
  --model_name_or_path google/gemma-3-1b-it \
  --train_file data/synthetic/annotations.0000.jsonl \
  --eval_file  data/synthetic/annotations.0001.jsonl \
  --output_dir ./gemma3-1b-qlora-ner \
  --max_seq_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bf16