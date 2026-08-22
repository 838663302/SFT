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


def process():
    datadict = load_dataset(
        "json",
        data_files=str(config.DATA_DIR / "generated_data.jsonl"),
    )
    # 划分数据集：validation 占比 0.5
    split = datadict["train"].train_test_split(test_size=0.5, seed=42)
    # train_test_split 默认把验证集命名为 "test"，重命名为 "validation"
    split["validation"] = split.pop("test")
    # 保存为 Arrow 格式（地址直接用 config.PROCESSED_DIR）
    split.save_to_disk(str(config.PROCESSED_DIR))
    print("train:", len(split["train"]), "条")
    print("validation:", len(split["validation"]), "条")


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

TOKENIZER_VERSION = "flan_t5_python_dict_v2_generated"


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
    # 与 main.py 保持一致：必须先加入 {} 这两个 token，
    # 否则 { / } 会被 tokenizer 映射成 <unk>，
    # 导致 get_dataset 中的 <unk> 检查直接报错。
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    num_added_tokens = tokenizer.add_tokens(["{", "}"])
    print("新增 token 数:", num_added_tokens)
    ds = get_dataset(tokenizer, is_train=True, max_samples=10)
    print(tokenizer.decode(ds[0]['labels'], skip_special_tokens=False))
    # print(ds[0]['labels'])
    # process()
