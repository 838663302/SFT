import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

MODEL_PATH = Path(__file__).parent.resolve() / "t5-json-final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("tokenizer vocab size:", len(tokenizer))
print("model vocab size:", model.get_input_embeddings().num_embeddings)

email = """
Good morning,Our customer Olivia Brown represents Acme Corporation. 
The company operates from Paris, UK. The primary contact email is olivia.brown@example.com. 
The phone number is +44 25 8475 1930.Kind regards,Michael BrownAccount Management
"""

instruction = """
Extract the detailed customer information from the following email and output as JSON format.

Email:
"""

prompt = instruction + email

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128,num_beams=4,do_sample=False,)

result = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("\n========== OUTPUT ==========")
print(result)
