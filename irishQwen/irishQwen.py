import torch
import torch.nn.functional as F
from statistics import mean
from tqdm import tqdm

from .qwen_transformer import QwenTransformer

from .embedding import EmbeddingLayer
from .linear_layer import LinearLayer
from .rotary_embedding import RotaryEmbedding
from .rms_norm import RMSNorm

class IrishQwen():
    """
    IrishChat rebuilt with the Qwen3 architecture.
    Defaults match Qwen3-0.6B.
    """
    block_class = QwenTransformer

    def reset_kv_cache(self):
        for block in self.transformer_blocks:
            block.reset_kv_cache()

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
        if n_experts != 0:
            raise ValueError("This teaching version supports dense Qwen3 only (n_experts=0).")
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

    def _forward_cached(self, token_ids, start_pos: int) -> torch.Tensor:
        """
        Run token_ids (list of ints) through the model, reading from and appending
        to every block's KV cache. The first token sits at absolute position start_pos.
        Returns next-token probabilities for the LAST position only: (1, 1, vocab_size).
        """
        if start_pos + len(token_ids) > self.ctx_len:
            raise ValueError(f"positions up to {start_pos + len(token_ids)} exceed ctx_len={self.ctx_len}")

        x_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        x = self.token_embedding.forward(x_ids)                 # (1, T, C)

        for block in self.transformer_blocks:
            x = block.forward_kv(x, start_pos)

        # Only the last position predicts the next token. Skipping the others
        # matters: the lm_head is (C x 151k), the most expensive matmul in the model.
        x = self.final_ln.forward(x[:, -1:, :])
        logits = self.lm_head.forward(x)
        return F.softmax(logits, dim=-1)

    def prefill(self, prompt_tokens) -> torch.Tensor:
        """
        Phase 1: process the whole prompt in ONE parallel forward pass.
        Fills every block's KV cache and returns the probs for the first new token.
        """
        self.reset_kv_cache()
        return self._forward_cached(prompt_tokens, start_pos=0)

    def decode_step(self, token_id: int, position: int) -> torch.Tensor:
        """
        Phase 2: process ONE new token, attending to everything in the cache.
        """
        return self._forward_cached([token_id], start_pos=position)

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
        Generate text using per-block KV cache.

        1) Prefill: run the whole prompt at once (fills KV cache).
        2) Decode: generate new tokens one at a time using cached K/V.
        """
        if len(prompt_tokens) == 0:
            raise ValueError("prompt_tokens must contain at least one token")

        # Keep only the rightmost context window.
        tokens = list(prompt_tokens[-self.ctx_len :])

        probs = self.prefill(tokens)
        current_len = len(tokens)

        for _ in range(max_new_tokens):
            next_probs = probs[0, -1, :]
            next_token = self.pick_next_token(
                next_probs, tokens, temperature, top_k, top_p, thinking_budget
            )

            tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

            # For simplicity, this educational version enforces ctx_len hard.
            # A production version would slide/truncate cache instead.
            if current_len >= self.ctx_len:
                break

            probs = self.decode_step(next_token, current_len)
            current_len += 1

        return tokens

    def softmax(self, X):
        """Normalize logits into probabilities over the vocabulary."""
        return F.softmax(X, dim=-1)

    def cross_entropy(self, Y_hat, Y):
        """Mean cross-entropy over non-padding tokens in (B, T, V) tensors."""

        log_probs = torch.log(Y_hat.clamp_min(1e-12))
        token_loss = -(Y * log_probs).sum(dim=2)  # (B, T)
        mask = (Y.argmax(dim=-1) != self.padding_idx).float()
        masked_loss = token_loss * mask
        normalizer = mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def get_accuracy(self, Y_hat, Y):
        """Fraction of correctly predicted non-padding tokens."""

        # labels: Y (B, T, V) one-hot
        Y_idx = Y.argmax(dim=-1)                # (B, T)
        preds = Y_hat.argmax(dim=-1)
        mask = (Y_idx != self.padding_idx).float()
        correct = (preds == Y_idx).float() * mask
        normalizer = mask.sum().clamp_min(1.0)
        return correct.sum() / normalizer

    def train(self, train_loader, epochs=100, learning_rate=0.001, verbose=True, optimizer=None):
        """Train over batches of token IDs and one-hot targets; return epoch loss and accuracy."""
        loss_history = []
        accuracy_history = []

        for i in tqdm(range(epochs), desc="Epochs"):
            batch_losses = []
            batch_accuracies = []

            for X_batch, Y_batch in tqdm(train_loader, desc="Batches"):
                # Forward propagation
                Y_hat_batch = self.forward(X_batch)

                # Calculate metrics for the whole epoch
                loss = self.cross_entropy(Y_hat_batch, Y_batch)
                accuracy = self.get_accuracy(Y_hat_batch, Y_batch)

                batch_losses.append(loss.item())
                batch_accuracies.append(accuracy.item())

                # Backward propagation
                self.backward(Y_hat_batch, Y_batch)

                # Update parameters
                if optimizer is not None:
                    optimizer.step()        # e.g. optimizers.Adam(model) -- see optimizers.py
                else:
                    self.update(learning_rate)

            loss_history.append(mean(batch_losses))
            accuracy_history.append(mean(batch_accuracies))

            if verbose and i % max(1, epochs // 10) == 0:
                print(f"Epoch {i+1}/{epochs}")
                print(f"loss: {loss_history[-1]:.5f}")
                print(f"accuracy: {accuracy_history[-1]:.5f}")
                print('Output test:')
                print("-" * 30)

        return {'loss_history': loss_history, 'accuracy_history': accuracy_history}
