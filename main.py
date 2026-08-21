# 注意：config 必须在 transformers/datasets 之前导入，确保缓存目录环境变量先生效
import config
import inspect
import subprocess

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import load_from_disk
from process import get_dataset

import os
import torch

# 兼容不同 transformers 版本的参数差异（新版本约 >=4.46，旧版本更早）：
#  - 按长度分组：新版本 train_sampling_strategy / 旧版本 group_by_length
#  - 评估策略：新版本 eval_strategy / 旧版本 evaluation_strategy
#  - Trainer 的分词器参数：新版本 processing_class / 旧版本 tokenizer
_args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
_USE_NEW_SAMPLING = "train_sampling_strategy" in _args_sig
_USE_NEW_EVAL = "eval_strategy" in _args_sig
_USE_REPORT_TO = "report_to" in _args_sig
_trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
_USE_PROCESSING_CLASS = "processing_class" in _trainer_sig
# 超参数
MODEL_NAME = "google/flan-t5-base"

MAX_TARGET_LENGTH = 256

class GPUUsageCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n========== GPU Usage ==========")
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
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
                        f"Memory: {mem_used.strip()} / {mem_total.strip()} MiB"
                    )
        print("===============================\n")

def main():
    # 1. 加载模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 3. 训练参数（针对 Kaggle 双 GPU 调整）
    # 说明：
    #  - Trainer 会自动检测多卡并用 DDP 分布式训练，无需额外代码。
    #  - per_device_* 是"每张卡"的 batch size，双卡全局 batch = 4 × 2 = 8。
    #  - 混合精度：fp16 在 T5 训练时会导致 loss 下溢为 0、grad_norm 为 nan（已实测），
    #    因此使用 fp16=False（fp32）。T4 虽支持 FP16 Tensor Core，但稳定性优先。
    #  - 按长度分组 / 评估策略参数在不同 transformers 版本名字不同，运行时会自动适配。

    # 基础参数（所有版本通用的稳定参数）
    # 显存说明：T4 单卡实际可用约 14.5GB，已开启 gradient_checkpointing 省显存，
    # 因此每卡 batch 可适度加大以提升 GPU 利用率（GPU 未跑满时可上调）。
    # 全局 batch = per_device_train_batch_size × GPU数 × gradient_accumulation_steps = 4×2×4 = 32
    kwargs = dict(
        output_dir=str(config.CHECKPOINT_DIR / "t5-json"),
        run_name="t5-json",
        num_train_epochs=3,
        per_device_train_batch_size=4,        # 每卡 4，双卡全局 8（GPU 未跑满，可上调）
        per_device_eval_batch_size=4,         # 每卡 4
        gradient_accumulation_steps=4,        # 累积后全局 batch = 4×2×4 = 32，训练稳定
        dataloader_num_workers=4,             # 多 worker 并行预取数据，缓解 CPU 取数压力
        dataloader_pin_memory=True,           # 锁页内存，加速 CPU→GPU 拷贝
        fp16=False,                            # fp16 会导致 T5 loss 下溢为 0，使用 fp32 训练
        max_grad_norm=1.0,
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=3e-5,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    # 梯度检查点（显著降低显存；属于长期稳定参数，新旧版本均存在）
    kwargs["gradient_checkpointing"] = True

    # 缓解 CPU 瓶颈：每个 worker 预取更多 batch，减少 GPU 等待数据的时间
    # （dataloader_prefetch_factor 是较新参数，旧版本不存在时跳过）
    if "dataloader_prefetch_factor" in _args_sig:
        kwargs["dataloader_prefetch_factor"] = 4

    # TensorBoard 日志：
    #  - 新版本用 report_to=["tensorboard"]，日志写到 output_dir/runs/<run_name>/
    #  - 旧版本用 logging_dir 显式指定日志目录
    if _USE_REPORT_TO:
        kwargs["report_to"] = ["tensorboard"]
    else:
        kwargs["logging_dir"] = str(config.LOG_DIR / "t5-json")

    # 按长度分组的参数：新版本用 train_sampling_strategy，旧版本用 group_by_length
    if _USE_NEW_SAMPLING:
        kwargs["train_sampling_strategy"] = "group_by_length"
        kwargs["length_column_name"] = "length"
    else:
        kwargs["group_by_length"] = True

    # 评估策略参数：新版本用 eval_strategy，旧版本用 evaluation_strategy
    if _USE_NEW_EVAL:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(**kwargs)

    # 4. 数据收集器
    # 注意：label_pad_token_id 必须用 -100（PyTorch CrossEntropyLoss 的 ignore_index），
    # 让标签中的 padding 位置不参与 loss 计算。若用 pad_token_id(0)，在 DDP 下会导致
    # loss 归零、grad_norm=nan。
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # 5. Trainer（分词器参数名随版本自适应：新版本 processing_class，旧版本 tokenizer）
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=get_dataset(tokenizer, is_train=True, 1000),  # 全量训练数据
        eval_dataset=get_dataset(tokenizer, is_train=False),
        data_collator=data_collator,
        callbacks=[GPUUsageCallback()],
    )
    if _USE_PROCESSING_CLASS:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)



    # 6. 训练并保存
    trainer.train()
    trainer.save_model(str(config.CHECKPOINT_DIR / "t5-json-final"))
    tokenizer.save_pretrained(str(config.CHECKPOINT_DIR / "t5-json-final"))



if __name__ == "__main__":
    main()
    # trainer = Seq2SeqTrainer(**trainer_kwargs)

    
