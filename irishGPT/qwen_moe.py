import torch
from .linear_layer import LinearLayer
from .swiglu import SwiGLU


class QwenSparseMoE:
    """
    Qwen3-MoE feed-forward: a router picks the top-k of E SwiGLU experts per token.

    Qwen3 router: softmax over ALL E experts, keep the top-k, renormalize so the
    k weights sum to 1. That is exactly the same as a softmax over just the top-k
    logits (the shared denominator cancels), which is what we compute here --
    the same trick as SparseMoE in sparse_moe.py.

    Differences from sparse_moe.py: experts are SwiGLU, router has no bias.

    Shapes:
      X:  (B, T, C)
      Y:  (B, T, C)
    """

    def __init__(self, model_dim, ffn_dim, n_experts=8, top_k=2, device="cpu"):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > n_experts:
            raise ValueError("top_k must be <= n_experts")

        self.device = device
        self.C = model_dim
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: token hidden state -> one logit per expert (no bias)
        self.router = LinearLayer(model_dim, n_experts, device=device, bias=False)

        self.experts = [SwiGLU(model_dim, ffn_dim, device=device) for _ in range(n_experts)]

    def forward(self, X):
        self.X_shape = X.shape
        bsz, seq_len, model_dim = X.shape
        bt = bsz * seq_len

        router_logits = self.router.forward(X)                                          # (B, T, E)

        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)  # (B, T, K)
        topk_probs = torch.softmax(topk_vals, dim=-1)                           # (B, T, K)

        self.topk_idx = topk_idx
        self.topk_probs = topk_probs

        x_flat = X.reshape(bt, model_dim)                 # (BT, C)
        idx_flat = topk_idx.reshape(bt, self.top_k)       # (BT, K)
        prob_flat = topk_probs.reshape(bt, self.top_k)    # (BT, K)

        y_flat = torch.zeros_like(x_flat)                 # (BT, C)

        # Cache assignment metadata for backward.
        self.assignment_cache = []

        for expert_id, expert in enumerate(self.experts):
            # All (token, slot) pairs where this expert was chosen.
            token_ids, slot_ids = (idx_flat == expert_id).nonzero(as_tuple=True)

            if token_ids.numel() == 0:
                self.assignment_cache.append(None)
                continue

            # SwiGLU expects 3D input, so present N tokens as (1, N, C).
            y_e = expert.forward(x_flat[token_ids].unsqueeze(0)).squeeze(0)  # (N, C)

            w_e = prob_flat[token_ids, slot_ids].unsqueeze(-1)  # (N, 1)
            y_flat.index_add_(0, token_ids, w_e * y_e)

            self.assignment_cache.append(
                {"token_ids": token_ids, "slot_ids": slot_ids, "y_e": y_e, "w_e": w_e}
            )

        return y_flat.reshape(bsz, seq_len, model_dim)

    def backward(self, dY):
        bsz, seq_len, model_dim = self.X_shape
        bt = bsz * seq_len

        dY_flat = dY.reshape(bt, model_dim)

        dX_flat = torch.zeros((bt, model_dim), device=dY.device, dtype=dY.dtype)
        d_topk_probs = torch.zeros_like(self.topk_probs.reshape(bt, self.top_k))  # (BT, K)

        for expert_id, expert in enumerate(self.experts):
            cache = self.assignment_cache[expert_id]
            if cache is None:
                continue

            token_ids = cache["token_ids"]
            slot_ids = cache["slot_ids"]

            # Y_token += w_e * y_e
            dY_e = dY_flat[token_ids] * cache["w_e"]                     # (N, C)
            d_w = (dY_flat[token_ids] * cache["y_e"]).sum(dim=-1)        # (N,)
            d_topk_probs[token_ids, slot_ids] += d_w

            dx = expert.backward(dY_e.unsqueeze(0)).squeeze(0)           # (N, C)
            dX_flat.index_add_(0, token_ids, dx)

        # Backprop through the top-k softmax: dvals = p * (g - sum(g*p))
        p = self.topk_probs.reshape(bt, self.top_k)
        dot = (d_topk_probs * p).sum(dim=-1, keepdim=True)
        d_topk_vals = p * (d_topk_probs - dot)  # (BT, K)

        # Scatter top-k gradients back into the full (BT, E) router logits.
        d_router_logits = torch.zeros((bt, self.n_experts), device=dY.device, dtype=dY.dtype)
        d_router_logits.scatter_add_(dim=1, index=self.topk_idx.reshape(bt, self.top_k), src=d_topk_vals)

        d_router_logits = d_router_logits.reshape(bsz, seq_len, self.n_experts)
        dX_flat += self.router.backward(d_router_logits).reshape(bt, model_dim)

        return dX_flat.reshape(bsz, seq_len, model_dim)

    def update(self, lr):
        self.router.update(lr)
        for expert, cache in zip(self.experts, self.assignment_cache):
            # Experts that saw no tokens this batch have no gradient.
            if cache is not None:
                expert.update(lr)

    def load_from_qwen3_tensors(self, gate, experts):
        """
        gate: (E, C) router weight
        experts: list of dicts with gate_proj / up_proj / down_proj weights
        """
        self.router.W = gate.to(self.device).contiguous()
        for expert, expert_state in zip(self.experts, experts):
            expert.load_from_qwen3_tensors(
                expert_state["gate_proj"], expert_state["up_proj"], expert_state["down_proj"]
            )
