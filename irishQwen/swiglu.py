from .linear_layer import LinearLayer
from .silu import SiLU


class SwiGLU:
    """
    Gated feed-forward network used by Qwen3 (and Llama, Mistral, ...).

    GPT-2 FFN:   Y = L2( GeLU( L1(X) ) )
    SwiGLU FFN:  Y = down( SiLU(gate(X)) * up(X) )

    The "up" branch carries the content, the "gate" branch decides (per hidden
    unit) how much of it to let through. No biases anywhere.

    Shapes: X (B, T, C) -> hidden (B, T, F) -> Y (B, T, C)
    """

    def __init__(self, model_dim, ffn_dim, device='cpu'):
        self.device = device

    def forward(self, X):
        pass

    def backward(self, dY):
        pass

    def update(self, lr):
        pass

    def load_from_qwen3_tensors(self, gate_proj, up_proj, down_proj):
        # HF nn.Linear weights are already (out, in) == LinearLayer.W. No transpose!
        self.gate.W = gate_proj.to(self.device).contiguous()
        self.up.W = up_proj.to(self.device).contiguous()
        self.down.W = down_proj.to(self.device).contiguous()
