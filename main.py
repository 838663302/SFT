from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
import config
from process import get_dataset
# 超参数
MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

class GPUUsageCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n========== GPU Usage ==========")

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True
        )

        for line in result.stdout.strip().split("\n"):
            gpu_id, gpu_util, mem_used, mem_total = line.split(",")

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
    #  - per_device_* 是"每张卡"的 batch size，双卡全局 batch = 8 × 2 = 16。
    #  - Kaggle GPU 为 T4（Turing 架构，支持 FP16 Tensor Core 加速）：
    #    * FP16 混合精度 → 开启（fp16=True），速度约提升 2 倍。
    #    * BF16 不支持（Turing 无 BF16 Tensor Core），故用 fp16 而非 bf16。
    #    * 双精度 FP64 在 T4 上被严重阉割且训练无用，绝不使用。
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR / "t5-json"),
        run_name="t5-json",
        num_train_epochs=3,
        per_device_train_batch_size=8,        # 每卡 8，双卡全局 16
        per_device_eval_batch_size=8,         # 每卡 8
        gradient_accumulation_steps=1,        # 双卡全局 batch 已够大，无需累积
        dataloader_num_workers=2,             # 加速数据加载
        fp16=True,                            # T4 用 FP16 混合精度加速（Tensor Core）
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        eval_strategy="epoch",
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

    # 4. 数据收集器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=model.config.pad_token_id, return_tensors="pt")

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=get_dataset(tokenizer, is_train=True, max_samples=10000),
        eval_dataset=get_dataset(tokenizer, is_train=False),
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[GPUUsageCallback()]
    )

    # 6. 训练并保存
    trainer.train()
    trainer.save_model(str(config.CHECKPOINT_DIR / "t5-json-final"))
    tokenizer.save_pretrained(str(config.CHECKPOINT_DIR / "t5-json-final"))


if __name__ == "__main__":
    main()
