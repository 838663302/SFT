# 注意：config 必须在 transformers/datasets 之前导入，确保缓存目录环境变量先生效
import config
import os
import re
import shutil
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# 需要过滤掉的字段（邮件数据专用）
REMOVE_COLUMNS = [
    "ID", "COUNTRY_CODE", "CREATED_BY", "CREATED_DATE",
    "LAST_UPDATED_BY", "LAST_UPDATED_DATE", "TENANT_ID",
]

# JSON 提取数据集的元数据字段
META_COLUMNS = ["topic", "title", "doc_style", "naming_convention", "tone"]


def process():
    datadict = load_dataset(
        "json",
        data_files=str(config.DATA_DIR / "generated_data.jsonl"),
    )
    # 划分数据集：train 90%，validation 10%
    split = datadict["train"].train_test_split(test_size=0.1, seed=42)
    # train_test_split 默认把验证集命名为 "test"，重命名为 "validation"
    split["validation"] = split.pop("test")
    # 保存为 Arrow 格式
    split.save_to_disk(str(config.PROCESSED_DIR))
    print("train:", len(split["train"]), "条")
    print("validation:", len(split["validation"]), "条")


def preprocess(batch, tokenizer):
    inputs = [
        f"{inst}: {text}" for inst, text in zip(batch["instruction"], batch["input"])
    ]
    outputs = batch["target"]
    enc = tokenizer(
        text=inputs,
        text_target=outputs,
        padding=False,
        truncation=True,
        max_length=512,
    )
    enc["length"] = [len(input) for input in enc["input_ids"]]
    return enc


# 程序启动时清理旧的 CACHE_DIR，避免历史缓存影响本次训练
# DDP 多进程环境下只在 rank 0 执行，避免并发删除冲突
if int(os.environ.get("RANK", "0")) == 0:
    if config.CACHE_DIR.exists():
        print(f"清理旧缓存目录: {config.CACHE_DIR}")
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)


# tokenizer/cache 版本标识（修改 tokenizer 或 preprocess 时需更新，避免复用旧缓存）
TOKENIZER_VERSION = "flan_t5_json_v3_standard"


def get_dataset(tokenizer, is_train=True, max_samples=None):
    split = "train" if is_train else "validation"
    dataset = load_from_disk(str(config.PROCESSED_DIR / split))

    # 默认只取前 max_samples 条（调试用），max_samples=None 使用全部数据
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    # 创建 cache 目录
    cache_dir = config.CACHE_DIR / "map_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 缓存文件名包含：split、tokenizer 版本、样本数、DDP rank
    local_rank = os.environ.get("LOCAL_RANK", "0")
    sample_tag = "all" if max_samples is None else f"n{max_samples}"
    cache_file = cache_dir / f"{split}.{TOKENIZER_VERSION}.{sample_tag}.rank{local_rank}.arrow"

    # 使用缓存或重新分词
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
            remove_columns=dataset.column_names,
            cache_file_name=str(cache_file),
        )

    # 检查 label 中是否包含 <unk>
    if len(dataset) > 0:
        sample = dataset[0]
        labels = sample["labels"]
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
        # 如果包含 <unk>，阻止训练
        if "<unk>" in decoded_label:
            raise RuntimeError(
                "\n"
                "检测到 label 中包含 <unk>！\n"
                "当前 tokenizer 与 tokenized dataset 不匹配，"
                "请检查 TOKENIZER_VERSION 或清理旧 cache。\n"
                f"cache_file = {cache_file}\n"
            )

    # DataCollatorForSeq2Seq 会负责生成 Tensor，不需要 set_format("torch")
    return dataset


if __name__ == "__main__":
    # 执行数据集划分与保存
    process()
    
    # 必须先加入 {} 这两个 token，否则 { / } 会被映射成 <unk>
    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # num_added_tokens = tokenizer.add_tokens(["{", "}"])
    # print("新增 token 数:", num_added_tokens)
    # ds = get_dataset(tokenizer, is_train=True, max_samples=10)
    # print("样本 0 Label 解码:")
    # print(tokenizer.decode(ds[0]['labels'], skip_special_tokens=False))
