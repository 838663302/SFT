from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import load_from_disk
import config
import inspect
import subprocess
from process import get_dataset

# 兼容不同 transformers 版本的参数差异（新版本约 >=4.46，旧版本更早）：
#  - 按长度分组：新版本 train_sampling_strategy / 旧版本 group_by_length
#  - 评估策略：新版本 eval_strategy / 旧版本 evaluation_strategy
#  - Trainer 的分词器参数：新版本 processing_class / 旧版本 tokenizer
_args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
_USE_NEW_SAMPLING = "train_sampling_strategy" in _args_sig
_USE_NEW_EVAL = "eval_strategy" in _args_sig
_trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
_USE_PROCESSING_CLASS = "processing_class" in _trainer_sig
# 超参数
MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
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
    # 0. 确保 processed 数据已生成（Kaggle 上首次运行时自动执行预处理）
    if not (config.PROCESSED_DIR / "train").exists():
        print("未找到 processed 数据，先执行 process() ...")
        from process import process
        process()
        print("process() 完成。")

    # 1. 加载模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 3. 训练参数（针对 Kaggle 双 GPU 调整）
    # 说明：
    #  - Trainer 会自动检测多卡并用 DDP 分布式训练，无需额外代码。
    #  - per_device_* 是"每张卡"的 batch size，双卡全局 batch = 8 × 2 = 16。
    #  - Kaggle GPU 为 T4（Turing 架构，支持 FP16 Tensor Core 加速）：
    #    * FP16 混合精度 → 开启（fp16=True），速度约提升 2 倍。
    #    * BF16 不支持（Turing 无 BF16 Tensor Core），故用 fp16 而非 bf16。
    #    * 双精度 FP64 在 T4 上被严重阉割且训练无用，绝不使用。
    #  - 按长度分组 / 评估策略参数在不同 transformers 版本名字不同，运行时会自动适配。

    # 基础参数（所有版本通用的稳定参数）
    kwargs = dict(
        output_dir=str(config.CHECKPOINT_DIR / "t5-json"),
        run_name="t5-json",
        num_train_epochs=3,
        per_device_train_batch_size=8,        # 每卡 8，双卡全局 16
        per_device_eval_batch_size=8,         # 每卡 8
        gradient_accumulation_steps=1,        # 双卡全局 batch 已够大，无需累积
        dataloader_num_workers=2,             # 加速数据加载
        fp16=True,                            # T4 用 FP16 混合精度加速（Tensor Core）
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=model.config.pad_token_id, return_tensors="pt")

    # 5. Trainer（分词器参数名随版本自适应：新版本 processing_class，旧版本 tokenizer）
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=get_dataset(tokenizer, is_train=True, max_samples=10000),
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
