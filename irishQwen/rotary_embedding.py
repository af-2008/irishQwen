import torch


class RotaryEmbedding:
    """
    Rotary Positional Embeddings (RoPE).

    Instead of ADDING a learned position vector to the token embedding (GPT-2),
    RoPE ROTATES each query/key vector by an angle that depends on its position.
    Pairs of dimensions (i, i + Dh/2) are treated as 2D points and rotated by
    angle  pos * theta^(-2i/Dh).  Because q·k only depends on the DIFFERENCE of
    the two rotation angles, attention scores end up depending on relative position.

    No trainable parameters -- just precomputed cos/sin tables, so one instance
    can be shared by every attention layer.
    """

    def __init__(self, head_dim, max_seq_len, theta=1_000_000.0, device="cpu"):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        # One frequency per dimension pair: (Dh/2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)        # (T_max, Dh/2)
        angles = torch.cat([freqs, freqs], dim=-1)      # (T_max, Dh)

        self.cos = angles.cos()
        self.sin = angles.sin()

    def _rotate_half(self, X):
        # (x1, x2) -> (-x2, x1): a 90 degree rotation of every dimension pair
        x1, x2 = X.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _tables(self, seq_len, start_pos):
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"position {start_pos + seq_len} exceeds max_seq_len={self.max_seq_len}")
        cos = self.cos[start_pos:start_pos + seq_len]   # (T, Dh), broadcasts over (B, H, T, Dh)
        sin = self.sin[start_pos:start_pos + seq_len]
        return cos, sin

    def forward(self, X, start_pos=0):
        """
        X: (B, H, T, Dh). start_pos is the absolute position of X[..., 0, :]
        (non-zero when decoding with a KV cache).
        """
        pass

    def backward(self, dY, start_pos=0):
        # A rotation's gradient is the rotation by the opposite angle.
        # (The transpose of rotate_half is -rotate_half.)
        pass

    def update(self, lr):
        # No trainable parameters.
        pass
