from datasets import load_dataset, load_from_disk
import config
from transformers import AutoTokenizer

# 需要过滤掉的字段（邮件数据专用）
REMOVE_COLUMNS = [
    "ID",
    "COUNTRY_CODE",
    "CREATED_BY",
    "CREATED_DATE",
    "LAST_UPDATED_BY",
    "LAST_UPDATED_DATE",
    "TENANT_ID",
]

# JSON 提取数据集的元数据字段
META_COLUMNS = ["topic", "title", "doc_style", "naming_convention", "tone"]


def process():
    # 从本地 parquet 文件加载 train 和 validation 数据集
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(config.DATA_DIR / "train-00000-of-00001.parquet"),
            "validation": str(config.DATA_DIR / "validation-00000-of-00001.parquet"),
        },
    )

    # 移除元数据字段（train 和 validation 统一处理）
    for split in dataset:
        dataset[split] = dataset[split].remove_columns(META_COLUMNS)

    # 保存为 Arrow 格式（保存整个 DatasetDict，保留 train/validation 结构）
    # Kaggle 上 PROCESSED_DIR 会指向可写的 /kaggle/working/data/processed
    dataset.save_to_disk(str(config.PROCESSED_DIR))

def preprocess(batch, tokenizer):
    # batched=True 时 batch 的每个字段是 list，需要逐元素拼接
    inputs = [
        f"{inst}: {text}"
        for inst, text in zip(batch["instruction"], batch["text"])
    ]
    outputs = batch["json"]
    # tokenizer 支持批量：text/text_target 传 list，返回长度一致的结果
    enc = tokenizer(
        text=inputs,
        text_target=outputs,
        padding=False,
        truncation=True,
    )
    enc['length'] = [len(input) for input in enc["input_ids"]]
    return enc

def get_dataset(tokenizer, is_train=True, max_samples=None):
    split = "train" if is_train else "validation"
    dataset = load_from_disk(str(config.PROCESSED_DIR / split))

    # 分词（batched=True，返回 input_ids / attention_mask / labels）
    dataset = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # 可选：抽样（用于快速测试）。max_samples 为 None 时返回全量数据
    if max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    dataset.set_format("torch")
    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    get_dataset(tokenizer, is_train=False)
