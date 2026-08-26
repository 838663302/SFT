# 注意：config 必须在 transformers/datasets 之前导入，确保缓存目录环境变量先生效
import config
import os
import inspect
import subprocess
import torch
import urllib3
import requests

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 禁用 SSL 验证（公司代理环境）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 禁用 FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from process import get_dataset

MODEL_NAME = "google/flan-t5-base"
# T5 原 tokenizer 对 { } 可能产生 <unk>，所以加入 tokenizer
ADDED_TOKENS = ["{", "}"]
MAX_TARGET_LENGTH = 256

# Transformers 版本兼容检测
_args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
_USE_NEW_SAMPLING = "train_sampling_strategy" in _args_sig
_USE_NEW_EVAL = "eval_strategy" in _args_sig
_USE_REPORT_TO = "report_to" in _args_sig
_trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
_USE_PROCESSING_CLASS = "processing_class" in _trainer_sig


class GPUUsageCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        print("\n========== GPU Usage ==========")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("nvidia-smi 不可用\n===============================\n")
            return
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            print("未检测到 GPU 信息")
        else:
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 4:
                    gpu_id, gpu_util, mem_used, mem_total = parts
                    print(f"GPU {gpu_id.strip()} | GPU Util: {gpu_util.strip()}% | Memory: {mem_used.strip()} / {mem_total.strip()} MiB")
        print("===============================\n")


def main():

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_added_tokens = tokenizer.add_tokens(ADDED_TOKENS)

    # 2. Model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
   
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight
            mean_emb = embeddings[:32100].mean(dim=0)
            std_emb = embeddings[:32100].std(dim=0)
            for token_name in ADDED_TOKENS:
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                embeddings[token_id] = mean_emb + torch.randn_like(mean_emb) * std_emb * 0.1
    model.config.use_cache = False
    
    # 3. 训练参数
    kwargs = dict(
        output_dir=str(config.CHECKPOINT_DIR / "t5-json"),
        run_name="t5-json",
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,  # 全局 batch = 4 × 2卡 × 1 = 8
        dataloader_num_workers=0 if not config.IS_KAGGLE else 4,
        dataloader_pin_memory=True,
        fp16=False,  # fp16 会导致 T5 loss=0, grad_norm=nan
        max_grad_norm=1.0,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        warmup_ratio=0.05,
        weight_decay=0.01,
        learning_rate=1e-4,
        save_total_limit=2,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    kwargs["gradient_checkpointing"] = True
    if "dataloader_prefetch_factor" in _args_sig:
        kwargs["dataloader_prefetch_factor"] = 4
    if _USE_REPORT_TO:
        kwargs["report_to"] = ["tensorboard"]
    else:
        kwargs["logging_dir"] = str(config.LOG_DIR / "t5-json")
    if _USE_NEW_SAMPLING:
        kwargs["train_sampling_strategy"] = "group_by_length"
        kwargs["length_column_name"] = "length"
    else:
        kwargs["group_by_length"] = True
    if _USE_NEW_EVAL:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    training_args = Seq2SeqTrainingArguments(**kwargs)

    # 4. Dataset
    print("\n========== LOAD DATASET ==========")
    train_dataset = get_dataset(tokenizer, is_train=True)
    eval_dataset = get_dataset(tokenizer, is_train=False)
    print("train size:", len(train_dataset))
    print("eval size:", len(eval_dataset))

    # 6. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
        label_pad_token_id=-100, return_tensors="pt",
    )

    # 7. Trainer
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator, callbacks=[GPUUsageCallback()],
    )
    if _USE_PROCESSING_CLASS:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    # 8. Train
    trainer.train()

    # 9. 保存模型（只让 rank0 写文件）
    final_dir = config.CHECKPOINT_DIR / "t5-json-final"
    if trainer.is_world_process_zero():
        final_dir.mkdir(parents=True, exist_ok=True)
        print("\n========== SAVE MODEL ==========")
        print("保存目录:", final_dir)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print("模型保存完成")
        print("\n文件列表:")
        for file in sorted(final_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"{file.name}: {size_mb:.2f} MB")
        print("================================\n")


if __name__ == "__main__":
    main()
