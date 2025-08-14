#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
You are an information extraction model for fictional, de-identified clinical notes.
Extract ONLY strict JSON conforming to this schema:

{
  "entities": [
    { 
      "mention": string, 
      "type":"Symptom|Diagnosis|Treatment", 
      "assertion":"Positive|Negated|Possible", 
      "context_window": string
    }
  ]
}

Rules:
 - Output JSON only, no extra keys or explanations.
 - Maintain medical plausibility; all content is fictional.
"""

def build_messages(data: Dict) -> List[Dict]:
    user = (
        "Extract entities with assertion from this note:\n\n"
        f"<NOTE>\n{data['text']}\n</NOTE>"
    )
    assistant = json.dumps(
        {"entities": data.get("entities", [])},
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

def target_modules_default():
    # for Llama/Gemma/Qwen layer
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", required=False)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=2e-5)

    # QLoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--num_proc", type=int, default=1, help="dataset map processes")
    args = ap.parse_args()

    # --------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------- Dataset (JSONL schema: text + entities[...]) ----------
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file
    ds = load_dataset("json", data_files=data_files)

    def render(example):
        messages = build_messages(example)
        # render using the model tokenizer
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(render, remove_columns=ds["train"].column_names, num_proc=args.num_proc)

    # --------- QLoRA: 4-bit Quantization + LoRA Adapter ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # QLoRA
        bnb_4bit_use_double_quant=True,   # Saving memory
        bnb_4bit_compute_dtype="bfloat16" if args.bf16 else "float16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # Turn on gradient checkpointing for saving VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules_default(),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # --------- Training ----------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps" if "eval" in ds else "no",
        eval_steps=1000,
        bf16=args.bf16,
        fp16=(not args.bf16),
        dataloader_pin_memory=False,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        args=training_args,
    )

    trainer.train()
    # save LoRA adapter (base weights are loaded as quantized, will not be overwritten)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()