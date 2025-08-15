from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "save/llama-3.2-1b-it-qlora",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# model.save_pretrained_gguf(
#   "save/llama-3.2-1b-it-4bit", 
#   tokenizer, 
#   quantization_method = "q4_k_m"
# )

model.save_pretrained_merged(
  "save/llama-3.2-1b-it-fp16", 
  tokenizer, 
  save_method = "merged_16bit",
)

# use vllm to serve
# vllm serve ./save/llama-3.2-1b-it-fp16