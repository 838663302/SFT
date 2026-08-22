# 注意：
# config 必须在 transformers/datasets 之前导入，
# 确保缓存目录环境变量先生效

import config
import os
import inspect
import subprocess
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from process import get_dataset

# ============================================================
# 基本配置
# ============================================================

MODEL_NAME = "google/flan-t5-base"

# 你的目标是结构化抽取，因此使用 Python dict 风格：
#
# {'retailPrice': 24.99}
#
# T5 原 tokenizer 对 { } 可能产生 <unk>，
# 所以这里把它们加入 tokenizer。

ADDED_TOKENS = ["{", "}"]
MAX_TARGET_LENGTH = 256

# ============================================================
# Transformers 版本兼容
# ============================================================

_args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
_USE_NEW_SAMPLING = "train_sampling_strategy" in _args_sig
_USE_NEW_EVAL = "eval_strategy" in _args_sig
_USE_REPORT_TO = "report_to" in _args_sig
_trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
_USE_PROCESSING_CLASS = "processing_class" in _trainer_sig

# ============================================================
# GPU 监控
# ============================================================


class GPUUsageCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 只让 rank0 打印，避免双卡重复输出
        if int(os.environ.get("RANK", "0")) != 0:
            return
        print("\n========== GPU Usage ==========")
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("nvidia-smi 不可用，跳过 GPU 监控")
            print("===============================\n")
            return
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            print("未检测到 GPU 信息")
        else:
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 4:
                    gpu_id, gpu_util, mem_used, mem_total = parts
                    print(
                        f"GPU {gpu_id.strip()} | "
                        f"GPU Util: {gpu_util.strip()}% | "
                        f"Memory: "
                        f"{mem_used.strip()} / "
                        f"{mem_total.strip()} MiB"
                    )
        print("===============================\n")


# ============================================================
# 主函数
# ============================================================


def main():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # print(
    #     f"[INIT] "
    #     f"RANK={rank}, "
    #     f"LOCAL_RANK={local_rank}, "
    #     f"WORLD_SIZE={world_size}, "
    #     f"CUDA={torch.cuda.current_device()}"
    # )
    # ========================================================
    # 1. Tokenizer
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("\n========== TOKENIZER ==========")
    print("原始 tokenizer size:", len(tokenizer))
    print("原始 { token:", tokenizer.convert_tokens_to_ids("{"))
    print("原始 } token:", tokenizer.convert_tokens_to_ids("}"))
    # --------------------------------------------------------
    # 加入 {}，避免被映射成 <unk>
    # --------------------------------------------------------
    num_added_tokens = tokenizer.add_tokens(ADDED_TOKENS)
    print("新增 token 数:", num_added_tokens)
    print("新的 tokenizer size:", len(tokenizer))
    print("{ token:", tokenizer.convert_tokens_to_ids("{"))
    print("} token:", tokenizer.convert_tokens_to_ids("}"))
    print("==============================\n")
    # ========================================================
    # 2. Model
    # ========================================================
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # tokenizer 增加 token 后，必须同步扩大 embedding
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    # T5 + gradient checkpointing
    model.config.use_cache = False
    print("模型参数量:", sum(p.numel() for p in model.parameters()))
    # ========================================================
    # 3. 训练参数
    # ========================================================
    kwargs = dict(
        output_dir=str(config.CHECKPOINT_DIR / "t5-json"),
        run_name="t5-json",
        num_train_epochs=3,
        # 每卡 batch
        per_device_train_batch_size=4,
        # 每卡 eval batch
        per_device_eval_batch_size=4,
        # 双卡：
        #
        # 4 × 2 × 4 = 32
        #
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # ----------------------------------------------------
        # T4 上你已经实际验证：
        #
        # fp16=True
        # -> loss=0
        # -> grad_norm=nan
        #
        # 因此这里暂时使用 FP32
        # ----------------------------------------------------
        fp16=False,
        # 梯度裁剪
        max_grad_norm=1.0,
        save_strategy="epoch",
        # 每个 epoch 做 generate evaluation
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        # ----------------------------------------------------
        # warmup
        # ----------------------------------------------------
        warmup_steps=500,
        weight_decay=0.01,
        # ----------------------------------------------------
        # 比之前 3e-5 稍高
        #
        # 你的任务是从预训练 T5 做结构化抽取，
        # 3e-5 对当前训练可能偏保守。
        # ----------------------------------------------------
        learning_rate=1e-4,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    # ========================================================
    # 4. Gradient Checkpointing
    # ========================================================
    kwargs["gradient_checkpointing"] = True
    # ========================================================
    # 5. DataLoader prefetch
    # ========================================================
    if "dataloader_prefetch_factor" in _args_sig:
        kwargs["dataloader_prefetch_factor"] = 4
    # ========================================================
    # 6. TensorBoard
    # ========================================================
    if _USE_REPORT_TO:
        kwargs["report_to"] = ["tensorboard"]
    else:
        kwargs["logging_dir"] = str(config.LOG_DIR / "t5-json")
    # ========================================================
    # 7. Length Grouping
    # ========================================================
    if _USE_NEW_SAMPLING:
        kwargs["train_sampling_strategy"] = "group_by_length"
        kwargs["length_column_name"] = "length"
    else:
        kwargs["group_by_length"] = True
    # ========================================================
    # 8. Evaluation Strategy
    # ========================================================
    if _USE_NEW_EVAL:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    training_args = Seq2SeqTrainingArguments(**kwargs)
    # ========================================================
    # 9. Dataset
    # ========================================================
    print("\n========== LOAD DATASET ==========")
    train_dataset = get_dataset(
        tokenizer,
        is_train=True,
    )
    eval_dataset = get_dataset(
        tokenizer,
        is_train=False,
    )
    print("train size:", len(train_dataset))
    print("eval size:", len(eval_dataset))
    print("columns:", train_dataset.column_names)
    # ========================================================
    # 10. 检查 tokenization
    #
    # 非常重要：
    # 确保你的 Python dict target
    # 不再出现 <unk>
    # ========================================================
    if rank == 0:
        print("\n========== TOKENIZATION CHECK ==========")
        sample = train_dataset[0]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        print("\nINPUT:")
        print(
            tokenizer.decode(
                input_ids,
                skip_special_tokens=False,
            )
        )
        # labels 中不存在 -100 时直接 decode
        valid_labels = [x for x in labels if x != -100]
        print("\nLABEL:")
        decoded_label = tokenizer.decode(
            valid_labels,
            skip_special_tokens=False,
        )
        print(decoded_label)
        print("\n是否包含 <unk>:", "<unk>" in decoded_label)
        print("\ninput token 数:", len(input_ids))
        print("label token 数:", len(valid_labels))
        print("========================================\n")
    # ========================================================
    # 11. Data Collator
    # ========================================================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        # padding label 不参与 loss
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    # ========================================================
    # 12. Trainer
    # ========================================================
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[GPUUsageCallback()],
    )
    if _USE_PROCESSING_CLASS:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    # ========================================================
    # 13. Train
    # ========================================================
    trainer.train()
    # ========================================================
    # 14. 保存最终模型
    #
    # 只让 rank0 写文件
    # ========================================================
    final_dir = config.CHECKPOINT_DIR / "t5-json-final"
    if trainer.is_world_process_zero():
        final_dir.mkdir(parents=True, exist_ok=True)
        print("\n========== SAVE MODEL ==========")
        print("保存目录:", final_dir)
        # 保存模型 + config
        trainer.save_model(str(final_dir))
        # 保存 tokenizer
        tokenizer.save_pretrained(str(final_dir))
        print("模型保存完成")
        # 打印最终文件
        print("\n文件列表:")
        for file in sorted(final_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"{file.name}: " f"{size_mb:.2f} MB")
        print("================================\n")


if __name__ == "__main__":
    main()
