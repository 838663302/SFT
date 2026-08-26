import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
ADDED_TOKENS = ["{", "}"]


def test_generate(model, tokenizer, text, tag):
    """
    测试 generate()
    """
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    # 模型可能在 GPU
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(f"\n========== {tag} GENERATE ==========")
    print(result)
    print("====================================\n")

    return result


def test_forward(model, tokenizer, text, tag):
    """
    测试 forward()，不经过 generate
    """
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    # 只传入 input_ids，避免传入模型不接受的参数
    input_ids = inputs["input_ids"].to(device)
    
    # T5 是 encoder-decoder 模型，forward() 需要 decoder_input_ids
    # generate() 会自动处理，但直接调用 forward() 需要手动提供
    batch_size = input_ids.shape[0]
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    logits = outputs.logits

    print(f"\n========== {tag} FORWARD ==========")
    print("logits shape:", tuple(logits.shape))
    print("logits max:", logits.max().item())
    print("logits min:", logits.min().item())
    print("logits mean:", logits.mean().item())
    print("logits std:", logits.std().item())

    # 最后一个位置的最大 token
    last_logits = logits[:, -1, :]
    next_token = last_logits.argmax(dim=-1)

    print("next token id:", next_token.item())
    print(
        "next token:",
        tokenizer.decode(
            [next_token.item()],
            skip_special_tokens=False
        )
    )
    print("====================================\n")

    return logits


def print_model_state(model, tokenizer, tag):
    """
    打印 tokenizer / model 当前词表状态
    """
    print(f"\n========== {tag} STATE ==========")

    print("tokenizer len:",
          len(tokenizer))

    print("tokenizer.vocab_size:",
          tokenizer.vocab_size)

    print("model.config.vocab_size:",
          model.config.vocab_size)

    print(
        "input embedding:",
        tuple(model.get_input_embeddings().weight.shape)
    )

    print(
        "output embedding:",
        tuple(model.get_output_embeddings().weight.shape)
    )

    print(
        "shared:",
        tuple(model.shared.weight.shape)
    )

    print(
        "lm_head:",
        tuple(model.lm_head.weight.shape)
    )

    print(
        "shared == lm_head:",
        model.shared.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )

    print(
        "decoder_start_token_id:",
        model.config.decoder_start_token_id
    )

    print(
        "pad_token_id:",
        model.config.pad_token_id
    )

    print(
        "eos_token_id:",
        model.config.eos_token_id
    )

    print(
        "generation_config.vocab_size:",
        getattr(
            model.generation_config,
            "vocab_size",
            None
        )
    )

    print("====================================\n")


def main():

    text = (
        "Translate English to German:\n"
        "The house is beautiful."
    )

    # ============================================================
    # 1. 原始模型
    # ============================================================

    print("\n\n")
    print("############################################################")
    print("# 1. ORIGINAL MODEL")
    print("############################################################")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    print_model_state(
        model,
        tokenizer,
        "ORIGINAL"
    )

    print(
        "原始 { token:",
        tokenizer.convert_tokens_to_ids("{")
    )

    print(
        "原始 } token:",
        tokenizer.convert_tokens_to_ids("}")
    )

    print(
        "原始 tokenize('{'):",
        tokenizer.tokenize("{")
    )

    print(
        "原始 tokenize('}'):",
        tokenizer.tokenize("}")
    )

    # 原始 generate
    test_generate(
        model,
        tokenizer,
        text,
        "ORIGINAL"
    )

    # 原始 forward
    logits_before_add = test_forward(
        model,
        tokenizer,
        text,
        "ORIGINAL"
    )

    # 记录原始 embedding
    original_embedding = (
        model.get_input_embeddings()
        .weight
        .detach()
        .clone()
    )

    original_vocab = (
        model.get_input_embeddings()
        .num_embeddings
    )

    print(
        "original model vocab:",
        original_vocab
    )

    # ============================================================
    # 2. 只 add_tokens，不 resize
    # ============================================================

    print("\n\n")
    print("############################################################")
    print("# 2. ADD TOKENS ONLY (NO RESIZE)")
    print("############################################################")

    num_added_tokens = tokenizer.add_tokens(
        ADDED_TOKENS
    )

    print(
        "新增 token 数:",
        num_added_tokens
    )

    print(
        "新的 tokenizer size:",
        len(tokenizer)
    )

    print(
        "{ token:",
        tokenizer.convert_tokens_to_ids("{")
    )

    print(
        "} token:",
        tokenizer.convert_tokens_to_ids("}")
    )

    print(
        "tokenize('{'):",
        tokenizer.tokenize("{")
    )

    print(
        "tokenize('}'):",
        tokenizer.tokenize("}")
    )

    # 注意：
    # 此时模型还没有 resize。
    # 先测试翻译是否仍然正常。

    test_generate(
        model,
        tokenizer,
        text,
        "AFTER ADD_TOKENS / BEFORE RESIZE"
    )

    logits_after_add = test_forward(
        model,
        tokenizer,
        text,
        "AFTER ADD_TOKENS / BEFORE RESIZE"
    )

    # 比较 add_tokens 是否影响原模型 forward
    add_forward_diff = (
        logits_before_add
        - logits_after_add
    ).abs()

    print(
        "\n========== ADD_TOKENS FORWARD DIFF =========="
    )

    print(
        "max diff:",
        add_forward_diff.max().item()
    )

    print(
        "mean diff:",
        add_forward_diff.mean().item()
    )

    print(
        "============================================\n"
    )

    # ============================================================
    # 3. resize
    # ============================================================

    print("\n\n")
    print("############################################################")
    print("# 3. RESIZE TOKEN EMBEDDINGS")
    print("############################################################")

    before_resize_embedding = (
        model.get_input_embeddings()
        .weight
        .detach()
        .clone()
    )

    # 关键：
    # 使用原始 model vocab + 新增 token
    new_vocab_size = (
        original_vocab
        + num_added_tokens
    )

    print(
        "original vocab:",
        original_vocab
    )

    print(
        "num added:",
        num_added_tokens
    )

    print(
        "target resize vocab:",
        new_vocab_size
    )

    model.resize_token_embeddings(
        new_vocab_size
    )

    print_model_state(
        model,
        tokenizer,
        "AFTER RESIZE"
    )

    after_resize_embedding = (
        model.get_input_embeddings()
        .weight
        .detach()
    )

    # ============================================================
    # 4. 检查 resize 是否修改原来的 embedding
    # ============================================================

    old_embedding_diff = (
        before_resize_embedding
        - after_resize_embedding[:original_vocab]
    ).abs()

    print(
        "\n========== OLD EMBEDDING CHECK =========="
    )

    print(
        "old embedding max diff:",
        old_embedding_diff.max().item()
    )

    print(
        "old embedding mean diff:",
        old_embedding_diff.mean().item()
    )

    print(
        "=========================================\n"
    )

    # ============================================================
    # 5. 检查新增 embedding
    # ============================================================

    new_rows = (
        after_resize_embedding[original_vocab:]
    )

    print(
        "\n========== NEW EMBEDDING =========="
    )

    print(
        "new rows shape:",
        tuple(new_rows.shape)
    )

    print(
        "mean:",
        new_rows.mean().item()
    )

    print(
        "std:",
        new_rows.std().item()
    )

    print(
        "max:",
        new_rows.abs().max().item()
    )

    print(
        "===================================\n"
    )

    # ============================================================
    # 6. resize 后，直接 forward
    # ============================================================

    logits_after_resize = test_forward(
        model,
        tokenizer,
        text,
        "AFTER RESIZE"
    )

    # 只比较原来的 vocabulary
    common_vocab = original_vocab

    resize_logits_diff = (
        logits_before_add[..., :common_vocab]
        - logits_after_resize[..., :common_vocab]
    ).abs()

    print(
        "\n========== RESIZE LOGITS DIFF =========="
    )

    print(
        "old vocab:",
        common_vocab
    )

    print(
        "max diff:",
        resize_logits_diff.max().item()
    )

    print(
        "mean diff:",
        resize_logits_diff.mean().item()
    )

    print(
        "========================================\n"
    )

    # ============================================================
    # 7. resize 后 generate
    # ============================================================

    test_generate(
        model,
        tokenizer,
        text,
        "AFTER RESIZE"
    )

    # ============================================================
    # 8. 把新增 embedding 归零
    # ============================================================

    print("\n\n")
    print("############################################################")
    print("# 4. ZERO NEW EMBEDDINGS")
    print("############################################################")

    with torch.no_grad():
        model.get_input_embeddings().weight[
            original_vocab:
        ] = 0

    print(
        "new embedding rows zeroed."
    )

    test_forward(
        model,
        tokenizer,
        text,
        "AFTER ZERO NEW EMBEDDINGS"
    )

    test_generate(
        model,
        tokenizer,
        text,
        "AFTER ZERO NEW EMBEDDINGS"
    )

    # ============================================================
    # 9. 最后检查 model / tokenizer 的状态
    # ============================================================

    print_model_state(
        model,
        tokenizer,
        "FINAL"
    )


if __name__ == "__main__":
    main()