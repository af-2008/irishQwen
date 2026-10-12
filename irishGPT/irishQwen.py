import torch
from .irishChat import IrishChat
from .embedding import EmbeddingLayer
from .linear_layer import LinearLayer
from .qwen_transformer import QwenTransformer
from .rotary_embedding import RotaryEmbedding
from .rms_norm import RMSNorm

class IrishQwen(IrishChat):
    """
    IrishChat rebuilt with the Qwen3 architecture.

    What changed from IrishChat (GPT-2):
      - No learned positional embedding: RoPE rotates Q and K inside attention
      - LayerNorm -> RMSNorm
      - Multi-head attention -> Grouped-Query Attention with QK-Norm, no biases
      - GeLU FFN -> SwiGLU FFN (or a sparse Mixture-of-Experts with n_experts > 0)

    The training loop and loss/accuracy (train, cross_entropy, get_accuracy,
    softmax) are inherited from IrishChat unchanged -- only the model body differs.

    Defaults match Qwen3-0.6B.
    """
    block_class = QwenTransformer   # IrishQwenKV swaps in a KV-cache block

    # Qwen3 special tokens for thinking mode
    THINK_START_ID = 151667         # <think>
    THINK_END_ID = 151668           # </think>

    # What we force when the thinking budget runs out. Just </think> works, but the
    # model was never trained on thoughts that stop mid-sentence and tends to keep
    # reasoning in its answer. Qwen suggests forcing a transition sentence first, e.g.
    #   model.budget_cutoff_tokens = tok.encode("\n\nConsidering the limited time by the user, "
    #       "I have to give the solution based on the thinking directly now.\n</think>\n\n")
    budget_cutoff_tokens = [THINK_END_ID]

    # Sampling settings recommended by the Qwen3 team, e.g. model.chat(p, **IrishQwen.THINKING_SAMPLING)
    # (Greedy decoding is discouraged in thinking mode: long traces fall into repetition loops.)
    THINKING_SAMPLING = dict(temperature=0.6, top_k=20, top_p=0.95)
    NON_THINKING_SAMPLING = dict(temperature=0.7, top_k=20, top_p=0.8)

    def __init__(
        self,
        vocab_size: int = 151936,
        ctx_len: int = 4096,
        d_model: int = 1024,
        n_layers: int = 28,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        ffn_dim: int = 3072,
        rope_theta: float = 1_000_000.0,
        n_experts: int = 0,
        top_k: int = 2,
        padding_idx: int = 151643,
    ):
        # Note: we deliberately don't call IrishChat.__init__ -- it would build GPT-2 layers.
        self.padding_idx = padding_idx
        # Remember the shape so save_converted_qwen3_checkpoint can rebuild this model
        self.config = dict(
            vocab_size=vocab_size, ctx_len=ctx_len, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, ffn_dim=ffn_dim,
            rope_theta=rope_theta, n_experts=n_experts, top_k=top_k, padding_idx=padding_idx,
        )
        self.ctx_len = ctx_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.token_embedding = EmbeddingLayer(vocab_size, d_model, device=self.device)
        self.rope = RotaryEmbedding(head_dim, ctx_len, theta=rope_theta, device=self.device)
        self.transformer_blocks = [
            self.block_class(
                d_model,
                rope=self.rope,
                ffn_dim=ffn_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                n_experts=n_experts,
                top_k=top_k,
                device=self.device,
            )
            for _ in range(n_layers)
        ]
        self.final_ln = RMSNorm(d_model, device=self.device)
        self.lm_head = LinearLayer(d_model, vocab_size, device=self.device, bias=False)

    @classmethod
    def qwen3_0_6b(cls, ctx_len=4096):
        return cls(ctx_len=ctx_len)

    @classmethod
    def from_checkpoint_config(cls, checkpoint_path, ctx_len=4096):
        """Build a model whose shape matches a converted checkpoint (see scripts/import_qwen3.py)."""
        config = dict(torch.load(checkpoint_path, map_location="cpu", mmap=True)["config"])
        return cls(ctx_len=min(ctx_len, config.pop("ctx_len")), **config)

    def load_converted_qwen3_checkpoint(self, checkpoint_path):
        """
        Load a project-native Qwen3 checkpoint (torch .pt) made by scripts/import_qwen3.py.

        Expected top-level keys:
          - embed_tokens: (V, C)
          - blocks: list[dict] with HF Qwen3 block keys
          - norm.weight: (C,)
          - optional lm_head.weight: (V, C)  (Qwen3-0.6B ties it to embed_tokens)
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        required = ["embed_tokens", "blocks", "norm.weight"]
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

        self.token_embedding.W = ckpt["embed_tokens"].T.to(self.device).contiguous()
        self.final_ln.gamma = ckpt["norm.weight"].to(self.device).contiguous()

        if len(ckpt["blocks"]) != len(self.transformer_blocks):
            raise ValueError(
                f"Block count mismatch: checkpoint has {len(ckpt['blocks'])}, model has {len(self.transformer_blocks)}"
            )

        for block, block_state in zip(self.transformer_blocks, ckpt["blocks"]):
            block.load_from_qwen3_block(block_state)

        if "lm_head.weight" in ckpt:
            self.lm_head.W = ckpt["lm_head.weight"].to(self.device).contiguous()
        else:
            # Small Qwen3 models use tied embeddings.
            self.lm_head.W = self.token_embedding.W.T.contiguous()

    def save_converted_qwen3_checkpoint(self, checkpoint_path):
        """
        Save weights in the same format load_converted_qwen3_checkpoint reads, so a
        trained model reloads with from_checkpoint_config + load_converted_qwen3_checkpoint.
        (torch.save(model) won't work: the forward caches hold unpicklable namedtuples.)
        """
        torch.save(
            {
                "config": self.config,
                "embed_tokens": self.token_embedding.W.T.cpu(),
                "norm.weight": self.final_ln.gamma.cpu(),
                "lm_head.weight": self.lm_head.W.cpu(),
                "blocks": [block.save_to_qwen3_block() for block in self.transformer_blocks],
            },
            checkpoint_path,
        )

    def forward(self, X, eval=False):
        """
        Args:
          x: (batch_size, seq_len) with integer token indices.
        Returns:
          probs: (batch_size, seq_len, vocab_size)
        """
        # Padding only exists in training batches. At inference (eval=True) we never mask,
        # otherwise a generated token that happens to equal padding_idx would be hidden.
        key_pad_mask = None if eval else (X == self.padding_idx)

        # No positional embedding here! Position enters through RoPE inside attention.
        X = self.token_embedding.forward(X)

        for block in self.transformer_blocks:
            X = block.forward(X, key_pad_mask=key_pad_mask)

        X = self.final_ln.forward(X)
        X = self.lm_head.forward(X)

        return self.softmax(X)

    def backward(self, Y_hat, Y):
        mask = (Y.argmax(dim=-1) != self.padding_idx).unsqueeze(-1).float()
        normalizer = mask.sum().clamp_min(1.0)
        dA = (Y_hat - Y) * mask / normalizer

        dA = self.lm_head.backward(dA)
        dA = self.final_ln.backward(dA)

        for block in reversed(self.transformer_blocks):
            dA = block.backward(dA)

        self.token_embedding.backward(dA)

    def update(self, lr):
        self.token_embedding.update(lr)
        for block in self.transformer_blocks:
            block.update(lr)
        self.final_ln.update(lr)
        self.lm_head.update(lr)

    def sample(self, next_probs, temperature=1.0, top_k=None, top_p=None):
        """
        Pick one token id from a (V,) probability vector.

        temperature: 0 = greedy (always pick most likely),
                     >0 = sample (higher = more random)
        top_k: only consider the k most likely tokens
        top_p: "nucleus" sampling -- only consider the smallest set of most likely
               tokens whose probabilities add up to at least top_p
        """
        if temperature == 0:
            return next_probs.argmax().item()

        # Scale probabilities by temperature and re-normalize
        probs = next_probs.pow(1.0 / temperature)
        probs = probs / probs.sum()

        # Sort once, most likely first; both filters cut off the tail of this list
        sorted_probs, sorted_ids = probs.sort(descending=True)

        if top_k is not None:
            sorted_probs = sorted_probs[:top_k]
            sorted_ids = sorted_ids[:top_k]
            sorted_probs = sorted_probs / sorted_probs.sum()

        if top_p is not None:
            # Keep a token if the tokens BEFORE it haven't reached top_p yet
            # (so the most likely token is always kept).
            cumulative = sorted_probs.cumsum(dim=0)
            keep = (cumulative - sorted_probs) < top_p
            sorted_probs = sorted_probs[keep]
            sorted_ids = sorted_ids[keep]
            sorted_probs = sorted_probs / sorted_probs.sum()

        choice = torch.multinomial(sorted_probs, 1)
        return sorted_ids[choice].item()

    def thinking_tokens_used(self, tokens):
        """
        How many tokens are inside a still-open <think> block (looking at the most
        recent <think>, whether the model wrote it or the prompt did).
        Returns None if there is none or it has already been closed.
        """
        if self.THINK_START_ID not in tokens:
            return None
        start = len(tokens) - 1 - tokens[::-1].index(self.THINK_START_ID)
        inside = tokens[start + 1:]
        return None if self.THINK_END_ID in inside else len(inside)

    def pick_next_token(self, next_probs, tokens, temperature, top_k, top_p, thinking_budget):
        """
        Sample the next token -- unless the model has used up its thinking budget,
        in which case we FORCE budget_cutoff_tokens (ending in </think>) one token
        per step. The model then has to write its answer from whatever reasoning
        it got done. Test-time compute, in one if-statement.
        """
        used = self.thinking_tokens_used(tokens)
        if thinking_budget is not None and used is not None and used >= thinking_budget:
            # The block stays open until the cutoff's final </think>, so "used"
            # tells us how far into the cutoff sequence we are.
            return self.budget_cutoff_tokens[used - thinking_budget]
        return self.sample(next_probs, temperature, top_k, top_p)

    def chat(
        self,
        prompt_tokens,
        max_new_tokens=200,
        temperature=1.0,
        eos_token_id=151645,
        top_k=None,
        top_p=None,
        thinking_budget=None,
    ):
        """
        Autoregressive generation.

        Args:
            prompt_tokens: list of int token IDs (from tokenizer.encode)
            max_new_tokens: maximum number of tokens to generate
            temperature: 0 = greedy (always pick most likely),
                         >0 = sample (higher = more random)
            eos_token_id: Qwen3 chat models end their turn with <|im_end|> (151645)
            top_k, top_p: optional sampling filters (see sample())
            thinking_budget: max tokens allowed inside <think> ... </think> (None = unlimited)
        Returns:
            list of int token IDs (prompt + generated)
        """
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            # Truncate to context window if the sequence is too long
            x = tokens[:, -self.ctx_len:]

            # Forward pass → probabilities (B, T, V)
            probs = self.forward(x, eval=True)

            # We only care about the LAST position's prediction
            next_probs = probs[0, -1, :]        # (V,)

            # Pick the next token
            next_token = self.pick_next_token(
                next_probs, tokens[0].tolist(), temperature, top_k, top_p, thinking_budget
            )

            # Append to the sequence
            tokens = torch.cat([
                tokens,
                torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            ], dim=1)

            # Stop when EOS is produced for the active tokenizer.
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return tokens[0].tolist()
