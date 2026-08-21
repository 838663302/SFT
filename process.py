# 注意：config 必须在 transformers/datasets 之前导入，确保缓存目录环境变量先生效
import config

import os
import re

from datasets import load_dataset, load_from_disk
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


def remove_newlines(example):
    # 去掉所有字符串字段中的 \n 以及 \n 后面的空格（对应 fix_json.py 的逻辑）
    for k, v in example.items():
        if isinstance(v, str):
            example[k] = re.sub(r'\n\s*', '', v)
    return example


def replace_quotes(example):
    # 将 json 字段中的双引号替换为单引号（对应 fix_quotes.py 的逻辑）
    if "json" in example:
        example["json"] = example["json"].replace('"', "'")
    return example


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

    # 数据清洗：去换行符及后续空格，json 字段双引号转单引号
    for split in dataset:
        dataset[split] = dataset[split].map(remove_newlines).map(replace_quotes)

    # 保存为 Arrow 格式（保存整个 DatasetDict，保留 train/validation 结构）
    # Kaggle 上 PROCESSED_DIR 会指向可写的 /kaggle/working/data/processed
    dataset.save_to_disk(str(config.PROCESSED_DIR))

    # 同步导出为 JSON 格式（每个 split 一个文件，供人工查看）
    for split in dataset:
        json_path = config.PROCESSED_DIR / f"{split}.json"
        dataset[split].to_json(str(json_path))
        print(f"已保存 {split} 至 {json_path}，共 {len(dataset[split])} 条")

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
    # 注意：datasets 5.x 中 Dataset[0:20] 切片返回的是普通 dict 而不是 Dataset，
    # 必须用 .select() 取子集，否则后续 .map() 会报 'dict' object has no attribute 'map'
    dataset = load_from_disk(str(config.PROCESSED_DIR / split))
    dataset = dataset.select(range(min(20, len(dataset))))

    # 分词（batched=True，返回 input_ids / attention_mask / labels）。
    # 关键：map 的缓存/临时文件默认写到源数据所在目录，而 Kaggle 上源数据在只读的
    # /kaggle/input 下，会报 "Read-only file system"。因此必须显式指定 cache_file_name
    # 写到可写目录（config.CACHE_DIR，Kaggle 上为 /kaggle/working/.cache）。
    # 首次运行生成缓存后，后续直接复用缓存，避免每次启动都重新全量分词拖慢训练。
    cache_dir = config.CACHE_DIR / "map_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 缓存文件名加上进程 rank，避免 DDP 多进程同时写同一个文件导致损坏
    local_rank = os.environ.get("LOCAL_RANK", "0")
    cache_file = cache_dir / f"{split}.arrow.rank{local_rank}"
    if cache_file.exists():
        print(f"复用 map 缓存: {cache_file}")
        from datasets import Dataset
        dataset = Dataset.from_file(str(cache_file))
        dataset = dataset.select(range(min(20, len(dataset))))
    else:
        dataset = dataset.map(
            lambda x: preprocess(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            cache_file_name=str(cache_file),
        )

    # 可选：抽样（用于快速测试）。max_samples 为 None 时返回全量数据
    if max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    # dataset.set_format("torch")
    return dataset


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(str(config.CHECKPOINT_DIR / "t5-json-final"))
    # ds = get_dataset(tokenizer, is_train=True, max_samples=10)
    # print(tokenizer.decode(ds[0]['labels'], skip_special_tokens=False))
    # print(ds[0]['labels'])
    process()
