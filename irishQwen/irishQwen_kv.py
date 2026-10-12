import torch
import torch.nn.functional as F

from .irishQwen import IrishQwen
from .qwen_transformer_kv_cache import QwenTransformerWithKVCache


class IrishQwenKV(IrishQwen):
    """
    IrishQwen with a very simple KV-cache generation path.

    Everything else (forward/backward/update/train/checkpoint loading) is
    inherited from IrishQwen -- only chat() changes.

    Generation is split into the two phases real inference servers use:
    - Prefill: the whole prompt in one parallel pass (compute-bound, fast per token)
    - Decode: one token per pass, reusing cached K/V (memory-bound, slow per token)

    Compared to IrishChatKV there is no positional embedding to look up per
    step: we just hand each block the token's absolute position and RoPE
    rotates the new query/key by that much.
    """
    block_class = QwenTransformerWithKVCache

    def reset_kv_cache(self):
        for block in self.transformer_blocks:
            block.reset_kv_cache()

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
            x = block.forward_step(x, start_pos)

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
        Generate text using per-block KV cache. Same arguments as IrishQwen.chat.

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
