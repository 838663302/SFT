import config
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---- Settings ----
model_path = "PaddlePaddle/PaddleOCR-VL-1.6"
image_path = "test.png"
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula' | 'spotting' | 'seal'
# ------------------

# ---- Image Preprocessing For Spotting ----
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
spotting_upscale_threshold = 1500

if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
    process_w, process_h = orig_w * 2, orig_h * 2
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    image = image.resize((process_w, process_h), resample_filter)

# ---------------------------

# -------- Inference --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# GPU（如 T4）用 fp16（图灵卡不支持 bfloat16）；CPU 无 fp16 加速，用 fp32
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

# 整模型单卡加载：1.6B @ fp16 仅占约 5~7GB，16GB 显卡足够。
# 注意不要用 device_map="auto" 多卡切分本模型——transformers 内置的
# PaddleOCR-VL 视觉塔在层被切分到不同卡时会触发设备不一致错误。
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=DTYPE,
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
print(result)
# ---------------------------
