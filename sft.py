###########################################################
# Load Data
###########################################################
import json
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset

dataset = load_dataset(
    "json", 
    data_files={
        "train": "data/synthetic/*.jsonl"
    }
)

def normalize(ex):
    ex["text"] = str(ex.get("text", ""))
    ents = ex.get("entities", [])
    if ents is None:
        ents = []

    if isinstance(ents, dict):
        ents = [ents]

    norm = []
    for e in ents:
        norm.append({
            "mention":        str(e.get("mention", "")),
            "type":           str(e.get("type", "")),
            "assertion":      str(e.get("assertion", "")),
            "context_window": str(e.get("context_window", "")),
        })
    ex["entities"] = norm
    return ex

dataset = dataset.map(normalize, desc="normalizing schema")
print(f'* loaded {len(dataset)} examples')

# split = ds["train"].train_test_split(test_size=0.1, seed=42)
# train_ds, eval_ds = split["train"], split["test"]


###########################################################
# Load Model
###########################################################
from unsloth import FastLanguageModel
import torch

# Choose any! We auto support RoPE Scaling internally!
max_seq_length = 2048
# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
dtype = None 
# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



###########################################################
# Prepare data
###########################################################
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

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

USER_PROMPT = """
Extract entities with assertion from this note:

{note}
"""

def formatting_prompts_func(examples):
    """
    Convert the examples into a format suitable for the model.
    """
    text_list = examples["text"]
    ents_list = examples["entities"]

    texts = []
    for note, ents in zip(text_list, ents_list):
        convo = [
            { "role": "system",    "content": SYSTEM_PROMPT },
            { "role": "user",      "content": USER_PROMPT.format(note=note) },
            { "role": "assistant", "content": json.dumps({"entities": ents}, ensure_ascii=False) }
        ]

        texts.append(
            tokenizer.apply_chat_template(
                convo, 
                tokenize = False, 
                add_generation_prompt = False
            )
        )
    
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

dataset = dataset['train']

# Train!

from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Start train!
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# save
path_model_ft = "save/llama-3.2-1b-it-qlora"
model.save_pretrained(path_model_ft)  # Local saving
tokenizer.save_pretrained(path_model_ft)