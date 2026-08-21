"""T5 JSON generation inference (interactive loop).
Type prompt or 'exit'/'quit' to quit.
"""

import json

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config

# Load model once
model_dir = config.CHECKPOINT_DIR / "t5-json-final"
print(f"Loading model from: {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Device: {device}")

while True:
    try:
        prompt = input("\nEnter prompt (instruction: text, or 'exit' to quit): ")
    except (EOFError, KeyboardInterrupt):
        break

    prompt = prompt.strip()
    if not prompt:
        continue
    if prompt.lower() in ("exit", "quit"):
        break

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"][0].tolist()
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)

    # Show all output tokens
    gen_ids = outputs[0].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("\n===== Output tokens =====")
    print(f"count: {len(gen_ids)}")
    for i, tid in enumerate(gen_ids):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        print(f"  [{i}] id={tid:6d}  token={piece!r}")

    print("\n===== Result =====")
    print(gen_text)
    print("=" * 40)

    # Parse to JSON object if possible
    try:
        obj = json.loads(gen_text)
        print("JSON object:")
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print("(generated text is not valid JSON)")
