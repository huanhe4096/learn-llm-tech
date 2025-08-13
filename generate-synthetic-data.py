#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ============ Unified Schema for Synthetic Data (for validation and documentation auto-generation) ============
class Entity(BaseModel):
    mention: str
    type: str  # "Symptom" | "Diagnosis" | "Treatment"
    assertion: str  # "Positive" | "Negated" | "Possible"
    context_window: str = ""

class Annotation(BaseModel):
    text: str
    entities: List[Entity] = Field(default_factory=list)

SYSTEM_PROMPT = (
    "You are a data generator for de-identified, fictional electronic health records.\n"
    "TASK: Create realistic yet entirely fictional clinical notes and extract entities.\n"
    "ENTITY TYPES: Symptom, Diagnosis, Treatment.\n"
    "ASSERTION: Positive (present), Negated (explicitly absent), Possible (suspected/likely/RO).\n"
    "RULES:\n"
    " - Notes must be medically plausible and fully fictional (no real PHI).\n"
    " - Aim ~{target_tokens} tokens per note (±20%). Use typical sections like HPI/ROS/Assessment/Plan where natural.\n"
    " - For each entity, provide a short context_window (5–12 words around the mention) copied from the note.\n"
    " - Balance assertion labels across the dataset over time; each batch should include some Negated/Possible.\n"
    "OUTPUT: Return ONLY strict JSON that conforms to the provided json_schema, with an array field `items`.\n"
)

USER_PROMPT_TEMPLATE = (
    "Generate {count} clinical annotations. Each annotation must follow this JSON unit schema:\n"
    "{unit_schema}\n"
    "IMPORTANT:\n"
    " - Do NOT include spans or any extra keys.\n"
    " - Use natural clinical language; include common abbreviations where appropriate (e.g., SOB, CP) but keep them clear in context.\n"
    " - Vary conditions and treatments (e.g., infections, cardiology, neurology, endocrine; meds/procedures/therapies).\n"
    "Return JSON with top-level key `items` (array of {count} objects)."
)

def estimate_safe_batch(count: int, tokens_per_annotation: int, cap: int = 16000) -> int:
    """Dynamically shrinks batch size based on estimated token output limit to avoid too large single requests.
    Uses a redundancy coefficient of 1.2; you can adjust based on model limits.
    """
    est = int(math.ceil(count * tokens_per_annotation * 1.2))
    if est <= cap:
        return count
    return max(1, cap // max(1, int(tokens_per_annotation * 1.2)))

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def next_shard_file(outdir: Path, shard_idx: int) -> Path:
    return outdir / f"annotations.{shard_idx:04d}.jsonl"

def write_jsonl_sharded(items: List[Dict[str, Any]], outdir: Path, shard_size: int = 1000) -> None:
    ensure_outdir(outdir)
    shard_idx = 0
    written = 0
    f = None
    try:
        for i, item in enumerate(items):
            if written % shard_size == 0:
                if f:
                    f.close()
                fpath = next_shard_file(outdir, shard_idx)
                f = open(fpath, "w", encoding="utf-8")
                shard_idx += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    finally:
        if f:
            f.close()

def build_unit_schema_str() -> str:
    # Simplified schema fragment for the model (reduces input tokens)
    return (
        '{\n'
        '  "text": "string",\n'
        '  "entities": [\n'
        '    {"mention":"string","type":"Symptom|Diagnosis|Treatment","assertion":"Positive|Negated|Possible","context_window":"string"}\n'
        '  ]\n'
        '}'
    )

@app.command()
def main(
  num: int = typer.Option(20, "--num", "-n", help="Total number of annotations to generate"),
  tokens_per_annotation: int = typer.Option(200, "--tokens-per-annotation", "-t", help="Target token count per annotation (approximate)"),
  batch_size: int = typer.Option(10, "--batch-size", "-b", help="Number of annotations generated per request"),
  model: str = typer.Option("gpt-5-nano", "--model", "-m", help="OpenAI model to use"),
  outdir: Path = typer.Option(Path("data/synthetic"), "--outdir", "-o", help="Output directory"),
  seed: Optional[int] = typer.Option(None, "--seed", help="Optional random seed (passed in prompt, not strict)"),
):
  """
  Generate synthetic EHR annotation data (JSONL, without spans) for debugging/testing fine-tuning pipeline.
  - Writes in volumes of 1000 lines/file: annotations.0000.jsonl, annotations.0001.jsonl ...
  - Application-layer batching: defaults to generating 20 items per request to reduce request count and prompt overhead.
  """
  load_dotenv()
  client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
  )

  ensure_outdir(outdir)

  unit_schema = build_unit_schema_str()
  total_needed = num
  all_items: List[Dict[str, Any]] = []

  batch_id = 0
  while total_needed > 0:
    want = min(batch_size, total_needed)
    safe = estimate_safe_batch(want, tokens_per_annotation, cap=16000)
    if safe < want:
      typer.echo(f"[warn] shrink batch from {want} to {safe} to respect output cap")
    count = safe

    system = SYSTEM_PROMPT.format(target_tokens=tokens_per_annotation)
    user = USER_PROMPT_TEMPLATE.format(count=count, unit_schema=unit_schema)
    if seed is not None:
      user += f"\nSEED: {seed + batch_id}"

    # [Reference: OpenAI Platform](https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com)
    resp = client.responses.create(
      model=model,
      input=[
        {"role": "assistant", "content": system},
        {"role": "user", "content": user},
      ],
      # Get max_output_tokens
      max_output_tokens=min(20000, int(tokens_per_annotation * count * 2)),
      reasoning={
        "effort": "low"
      }
    )

    # SDK provides output_text; for finer control, iterate over output/choices. [Reference: OpenAI Platform](https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com)
    text = getattr(resp, "output_text", None) or json.dumps(getattr(resp, "output", {}), default=lambda o: getattr(o, "__dict__", str(o)))
    
    print(f"[debug] raw response: {text[:1000]}...")  # Print first 1000 chars for debugging
    try:
      data = json.loads(text)
    except Exception:
      # Fallback: in some cases need to get from structured output fields (SDK version differences)
      # You can also print resp to debug the structure.
      raise RuntimeError("Model did not return valid JSON. Enable logging to inspect raw response.")

    items = data.get("items", [])
    if not isinstance(items, list) or len(items) != count:
      typer.echo(f"[warn] expected {count} items, got {len(items)}; continuing.")

    # Pydantic validation + normalization to ensure consistent output
    clean: List[Dict[str, Any]] = []
    for it in items:
        obj = Annotation(**it)  # Validate fields and types
        # clean.append(json.loads(obj.json(ensure_ascii=False)))
        clean.append(obj.model_dump())

    all_items.extend(clean)
    total_needed -= len(clean)
    batch_id += 1
    typer.echo(f"[ok] batch#{batch_id}: accepted {len(clean)} annotations; remaining {total_needed}")

  # Write in volumes, 1000 lines per file
  write_jsonl_sharded(all_items, outdir, shard_size=1000)
  typer.echo(f"Done. Wrote {len(all_items)} annotations into {outdir}")
    
if __name__ == "__main__":
    app()