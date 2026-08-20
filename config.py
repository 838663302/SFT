from pathlib import Path

# ============================================================
# 环境判断：线上（Kaggle） / 线下（本地）
# ============================================================
IS_KAGGLE = Path('/kaggle/input').exists()

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
    PROCESSED_DIR = DATA_DIR / "data" / "processed"

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
