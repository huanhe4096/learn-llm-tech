from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "save/llama-3.2-1b-it-qlora",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)


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

messages = [
  { "role": "system", "content": SYSTEM_PROMPT },
  { "role" : "user", "content" : USER_PROMPT.format(note="Received shot in left arm. About one hour after receiving the shot, I started getting very severe pain in my upper right arm that slowly spread to the right side of my neck. My right ear began to feel slightly funny, but no pain. Was going to the hospital but the all the pain went away after about 15 to 20 minutes. No problems after that so I didn't go to the hospital. I did not take any medication.")}
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

print(_)