from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data/models/gemma3-3-1b-it-qlora",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role" : "system", "content" : "Given an incomplit set of chess moves and the game's final score, write the last missing chess move.\n\nInput Format: A comma-separated list of chess moves followed by the game score.\nOutput Format: The missing chess move"},
    {"role": "user", "content": '{"moves": ["c2c4", "g8f6", "b1c3", "c7c5", "g1f3", "e7e6", "e2e3", "d7d5", "d2d4", "b8c6", "c4d5", "e6d5", "f1e2", "c5c4", "c1d2", "f8b4", "a1c1", "e8g8", "b2b3", "b4a3", "c1b1", "c8f5", "b3c4", "f5b1", "d1b1", "d5c4", "e2c4", "a3b4", "e1g1", "a8c8", "f1d1", "d8a5", "c3e4", "f6e4", "b1e4", "b4d2", "f3d2", "c8c7", "d2f3", "c6b8", "c4b3", "b8d7", "e4f4", "c7c3", "e3e4", "a5b5", "e4e5", "a7a5", "f4e4", "a5a4", "b3d5", "h7h6", "d1b1", "b5d3", "e4d3", "c3d3", "e5e6", "d7f6", "e6f7", "g8h7", "d5e6", "g7g6", "h2h4", "f6e4", "b1b7", "h7g7", "b7a7", "d3d1", "g1h2", "e4f2", "a7a4", "d1h1", "h2g3", "f2e4", "g3f4", "e4d6", "f3e5", "h1h4", "f4e3", "d6f5", "e3d3", "f8d8", "e5d7", "h4g4", "f7f8b", "d8f8", "d7f8", "g7f8", "e6d5", "g4g3", "d3e4", "g3g2", "e4e5", "g2d2", "a4a8", "f8e7", "a8a7", "e7d8", "d5e6", "d2e2", "e5f6", "e2f2", "a7d7", "d8e8", "f6g6", "f5h4", "g6g7", "f2g2", "g7h7", "h4f3", "h7h6", "g2a2", "d4d5", "a2h2", "h6g7", "f3g5", "g7f6", "g5e4", "f6e5", "e4c3", "d7b7", "h2e2", "e5d4", "c3d5", "d4d5", "e2d2", "d5e5", "d2f2", "e6d5", "e8f8", "d5e6", "f2h2", "b7c7", "h2h5", "e6f5", "f8e8", "e5d6", "h5h6", "f5e6", "e8f8", "c7f7", "f8g8", "d6e5", "h6g6", "e5d5", "g8h8", "d5d6", "g6g5", "d6e7", "g5g7", "e7f6", "g7f7", "?"], "result": "1/2-1/2"}'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = True, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024, # Increase for longer outputs!
    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)