import os
from pathlib import Path

# ============================================================
# 环境判断：线上（Kaggle） / 线下（本地）
# ============================================================
IS_KAGGLE = Path('/kaggle/input').exists()

# 缓存根目录：必须可写（Kaggle 默认 ~/.cache 可能不可写，显式指定到可写位置）
CACHE_DIR = Path('/kaggle/working/.cache') if IS_KAGGLE else Path(__file__).parent.resolve() / ".cache"

# 禁用 xet 下载协议（新版 huggingface_hub 默认启用，走 AWS CDN 分块下载，
# 在代理环境下常报 403 Forbidden / 连接中断，导致模型权重下载失败）。
# 回退到传统 HTTP 下载更稳定。必须在导入 huggingface_hub/transformers 前生效。
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 设置 Hugging Face 相关缓存（必须在导入 transformers/datasets 前生效）
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "huggingface" / "datasets")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface" / "transformers")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")

# 确保缓存目录存在且可写（否则 datasets.map() 写临时文件会失败）
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if IS_KAGGLE:
    # ===== 线上（Kaggle）=====
    # 输入数据目录（只读）
    DATA_DIR = Path('/kaggle/input/datasets/xiaonanhaiaichixigua/extractjson/data')
    # 输出目录（必须写到 /kaggle/working，可写）
    OUTPUT_DIR = Path('/kaggle/working')
    # 预处理后数据保存目录（process.py 生成）
    PROCESSED_DIR = DATA_DIR / "processed"
else:
    # ===== 线下（本地）=====
    BASE_DIR = Path(__file__).parent.resolve()
    # 输入数据目录（项目根目录下的 data）
    DATA_DIR = BASE_DIR / "data"
    # 输出目录（项目根目录，可写）
    OUTPUT_DIR = BASE_DIR
    # 预处理后数据保存目录（process.py 生成）
    PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================
# 输入数据文件（原始 parquet）
# ============================================================
TRAIN_FILE = DATA_DIR / "train-00000-of-00001.parquet"
VALIDATION_FILE = DATA_DIR / "validation-00000-of-00001.parquet"
# 单个数据文件路径
DATA_FILE = DATA_DIR / "data.csv"

# ============================================================
# 输出目录（processed / checkpoints / log）
# ============================================================

# 检查点保存目录
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# 日志目录
LOG_DIR = OUTPUT_DIR / "log"
