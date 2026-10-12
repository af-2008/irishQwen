import argparse
import torch


BLOCK_KEYS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
]
DENSE_MLP_KEYS = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


def convert_qwen3_state_dict(state, config):
    """
    HF Qwen3 / Qwen3-MoE state_dict -> project-native checkpoint dict.
    Unlike GPT-2, HF Qwen3 uses nn.Linear, whose (out, in) weights already match
    our X @ W.T convention, so nothing needs transposing.
    """
    n_experts = getattr(config, "num_experts", 0) or 0

    converted = {
        "config": {
            "vocab_size": config.vocab_size,
            "ctx_len": config.max_position_embeddings,
            "d_model": config.hidden_size,
            "n_layers": config.num_hidden_layers,
            "n_heads": config.num_attention_heads,
            "n_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "ffn_dim": config.moe_intermediate_size if n_experts else config.intermediate_size,
            "rope_theta": config.rope_parameters["rope_theta"] if hasattr(config, "rope_parameters") else config.rope_theta,
            "n_experts": n_experts,
            "top_k": getattr(config, "num_experts_per_tok", 2),
        },
        "embed_tokens": state["model.embed_tokens.weight"].detach().float().cpu(),   # (V, C)
        "norm.weight": state["model.norm.weight"].detach().float().cpu(),            # (C,)
        "blocks": [],
    }
    if not config.tie_word_embeddings:
        converted["lm_head.weight"] = state["lm_head.weight"].detach().float().cpu()  # (V, C)

    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        block = {k: state[prefix + k].detach().float().cpu() for k in BLOCK_KEYS}

        if n_experts:
            block["mlp.gate.weight"] = state[prefix + "mlp.gate.weight"].detach().float().cpu()  # (E, C)
            block["mlp.experts"] = [_expert_weights(state, prefix, e) for e in range(n_experts)]
        else:
            block.update({k: state[prefix + k].detach().float().cpu() for k in DENSE_MLP_KEYS})

        converted["blocks"].append(block)

    return converted


def _expert_weights(state, prefix, e):
    # Older transformers: one nn.Linear per expert
    if prefix + f"mlp.experts.{e}.gate_proj.weight" in state:
        return {
            name: state[prefix + f"mlp.experts.{e}.{name}.weight"].detach().float().cpu()
            for name in ["gate_proj", "up_proj", "down_proj"]
        }
    # transformers >= 5: all experts fused into 3D tensors
    gate, up = state[prefix + "mlp.experts.gate_up_proj"][e].detach().float().cpu().chunk(2, dim=0)
    down = state[prefix + "mlp.experts.down_proj"][e].detach().float().cpu()
    return {"gate_proj": gate, "up_proj": up, "down_proj": down}


def main():
    parser = argparse.ArgumentParser(description="Convert HF Qwen3 to project-native checkpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model id (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--out", default="Datasets/qwen3_0.6b_converted.pt", help="Output .pt file path")
    parser.add_argument(
        "--tokenizer_dir",
        default="Datasets/qwen3_tokenizer_assets",
        help="Directory to save vocab.json, merges.txt, and tokenizer_config.json",
    )
    args = parser.parse_args()

    # Conversion-time dependency only.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    converted = convert_qwen3_state_dict(model.state_dict(), model.config)

    torch.save(converted, args.out)
    print(f"Saved converted checkpoint to {args.out}")
    tokenizer.save_pretrained(args.tokenizer_dir)
    print(f"Saved tokenizer assets to {args.tokenizer_dir}")


if __name__ == "__main__":
    main()
