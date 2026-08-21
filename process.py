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
            example[k] = re.sub(r"\n\s*", "", v)
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
        f"{inst}: {text}" for inst, text in zip(batch["instruction"], batch["text"])
    ]
    outputs = batch["json"]
    # tokenizer 支持批量：text/text_target 传 list，返回长度一致的结果
    enc = tokenizer(
        text=inputs,
        text_target=outputs,
        padding=False,
        truncation=True,
    )
    enc["length"] = [len(input) for input in enc["input_ids"]]
    return enc


# ============================================================
# tokenizer/cache 版本
#
# 每次修改 tokenizer、增加 token、修改 preprocess 时，
# 改这个版本号，避免错误复用旧的 Arrow cache。
# ============================================================

TOKENIZER_VERSION = "flan_t5_python_dict_v1"


def get_dataset(tokenizer, is_train=True, max_samples=20):
    # --------------------------------------------------------
    # 1. 加载原始 Dataset
    # --------------------------------------------------------
    split = "train" if is_train else "validation"
    dataset = load_from_disk(str(config.PROCESSED_DIR / split))
    # --------------------------------------------------------
    # 2. 默认只取前 20 条（调试/快速验证用）
    #
    # max_samples=20:
    #     默认，只使用前 20 条
    #
    # max_samples=None:
    #     使用全部数据
    # --------------------------------------------------------
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
    # --------------------------------------------------------
    # 3. 创建 cache 目录
    # --------------------------------------------------------
    cache_dir = config.CACHE_DIR / "map_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------
    # 4. DDP rank
    #
    # 每个 GPU 使用自己的 cache 文件，
    # 避免多个进程同时写同一个 Arrow 文件。
    # --------------------------------------------------------
    local_rank = os.environ.get("LOCAL_RANK", "0")
    # --------------------------------------------------------
    # 5. max_samples 也加入 cache 文件名
    #
    # 防止：
    #
    # 20 条数据的 cache
    #     ↓
    # 后面被错误当成全量数据使用
    # --------------------------------------------------------
    if max_samples is None:
        sample_tag = "all"
    else:
        sample_tag = f"n{max_samples}"
    # --------------------------------------------------------
    # 6. tokenizer 版本加入 cache 文件名
    #
    # 你这次新增：
    #
    # {
    # }
    #
    # 所以一定不能继续读取以前 tokenizer 产生的 cache。
    # --------------------------------------------------------
    cache_file = cache_dir / (
        f"{split}." f"{TOKENIZER_VERSION}." f"{sample_tag}." f"rank{local_rank}.arrow"
    )
    # --------------------------------------------------------
    # 7. 使用 cache
    # --------------------------------------------------------
    if cache_file.exists():
        print(f"复用 map 缓存: {cache_file}")
        from datasets import Dataset

        dataset = Dataset.from_file(str(cache_file))
    else:
        print(f"重新执行 tokenizer: {split}")
        print(f"cache: {cache_file}")
        dataset = dataset.map(
            lambda x: preprocess(x, tokenizer),
            batched=True,
            # 原来的 instruction/text/json 等字段
            # tokenizer 后不再需要
            remove_columns=dataset.column_names,
            cache_file_name=str(cache_file),
        )
    # --------------------------------------------------------
    # 8. 检查 label
    #
    # 特别检查 <unk>
    #
    # 你的 tokenizer 已经新增：
    #
    # { -> 32100
    # } -> 32101
    #
    # 因此正常情况下 Python dict target
    # 不应该再出现 <unk>。
    # --------------------------------------------------------
    if len(dataset) > 0:
        sample = dataset[0]
        labels = sample["labels"]
        # 去掉 padding 的 -100
        valid_labels = [x for x in labels if x != -100]
        decoded_label = tokenizer.decode(valid_labels, skip_special_tokens=False)
        print("\n========== DATASET CHECK ==========")
        print("split:", split)
        print("dataset size:", len(dataset))
        print("tokenizer size:", len(tokenizer))
        print("'{' token id:", tokenizer.convert_tokens_to_ids("{"))
        print("'}' token id:", tokenizer.convert_tokens_to_ids("}"))
        print("label token count:", len(valid_labels))
        print("decoded label:")
        print(decoded_label)
        print("contains <unk>:", "<unk>" in decoded_label)
        print("==================================\n")
        # ----------------------------------------------------
        # 如果已经确定 Python dict target 不应该包含 <unk>，
        # 直接阻止训练，避免再次浪费几个小时。
        # ----------------------------------------------------
        if "<unk>" in decoded_label:
            raise RuntimeError(
                "\n"
                "检测到 label 中包含 <unk>！\n"
                "当前 tokenizer 与 tokenized dataset 不匹配，"
                "请检查 TOKENIZER_VERSION 或清理旧 cache。\n"
                f"cache_file = {cache_file}\n"
            )
    # --------------------------------------------------------
    # 9. 不需要 set_format("torch")
    #
    # DataCollatorForSeq2Seq 会负责生成 Tensor。
    # --------------------------------------------------------
    return dataset


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(str(config.CHECKPOINT_DIR / "t5-json-final"))
    # ds = get_dataset(tokenizer, is_train=True, max_samples=10)
    # print(tokenizer.decode(ds[0]['labels'], skip_special_tokens=False))
    # print(ds[0]['labels'])
    process()
