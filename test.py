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

# 非 spotting 任务：限制图像长边，控制视觉 token 数，避免大图撑爆显存
if task != "spotting":
    MAX_SIDE = 768  # 长边上限；OCR/表格在此分辨率下已足够
    w, h = image.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"图像已缩放: {orig_w}x{orig_h} -> {image.size[0]}x{image.size[1]}")

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
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 加速 matmul
    torch.backends.cudnn.allow_tf32 = True

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=DTYPE,
    attn_implementation="sdpa",  # 用 PyTorch SDPA 高效 attention，避免 eager 模式逐步全量计算
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path)

if DEVICE == "cuda":
    torch.cuda.empty_cache()  # 释放加载过程中的缓存碎片
    print(f"模型加载后显存 allocated/reserved: {torch.cuda.memory_allocated()/1e9:.2f} / {torch.cuda.memory_reserved()/1e9:.2f} GB")

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

if DEVICE == "cuda":
    torch.cuda.empty_cache()
    print(f"图像 patch 数: {inputs['pixel_values'].shape[0]}")
    print(f"推理前显存 allocated/reserved: {torch.cuda.memory_allocated()/1e9:.2f} / {torch.cuda.memory_reserved()/1e9:.2f} GB")

outputs = model.generate(**inputs, max_new_tokens=2048)  # 足够覆盖完整 OCR 结果；模型会在 EOS 时自动停止
result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
print(result)
# ---------------------------
