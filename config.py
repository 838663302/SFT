from pathlib import Path

# Kaggle 环境检测：输入数据在 /kaggle/input（只读），输出必须写到 /kaggle/working
if Path('/kaggle/input').exists():
    # 数据集实际挂载路径（线上实测）
    BASE_DIR = Path('/kaggle/input/datasets/xiaonanhaiaichixigua/extractjson')
    OUTPUT_DIR = Path('/kaggle/working')
else:
    # 本地运行：项目根目录
    BASE_DIR = Path(__file__).parent.resolve()
    OUTPUT_DIR = BASE_DIR

# ===== 输入数据（只读区，Kaggle 上即 /kaggle/input）=====
# 原始 parquet 数据所在目录。
# 优先使用 <BASE_DIR>/data（与本地 data/ 结构一致）；
# 若该目录下没有 parquet，则自动回退到 BASE_DIR 根目录查找，
# 以兼容 Kaggle 上传时目录层级不同的情况。
def _find_data_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data",
        base_dir,
    ]
    for cand in candidates:
        try:
            if any(cand.glob("*.parquet")):
                return cand
        except OSError:
            continue
    # 都找不到时，默认返回 <BASE_DIR>/data（与本地结构一致）
    return base_dir / "data"


DATA_DIR = _find_data_dir(BASE_DIR)
# 单个数据文件路径
DATA_FILE = DATA_DIR / "data.csv"

# ===== 输出目录（Kaggle 上必须写到 /kaggle/working）=====
# 预处理后数据保存目录（process.py 生成）
PROCESSED_DIR = OUTPUT_DIR / "data" / "processed"
# 检查点保存目录
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# 日志目录
LOG_DIR = OUTPUT_DIR / "log"
