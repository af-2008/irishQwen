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

        self.gate = LinearLayer(model_dim, ffn_dim, device, bias=False)
        self.up   = LinearLayer(model_dim, ffn_dim, device, bias=False)
        self.act  = SiLU()
        self.down = LinearLayer(ffn_dim, model_dim, device, bias=False)

    def forward(self, X):
        self.g = self.act.forward(self.gate.forward(X))     # (B, T, F)
        self.u = self.up.forward(X)                          # (B, T, F)

        return self.down.forward(self.g * self.u)

    def backward(self, dY):
        dH = self.down.backward(dY)                          # (B, T, F)

        # Elementwise gate:  H = g * u
        dg = dH * self.u
        du = dH * self.g

        # X fed both branches, so its gradient is the sum of both
        dX = self.gate.backward(self.act.backward(dg)) + self.up.backward(du)
        return dX

    def update(self, lr):
        self.gate.update(lr)
        self.up.update(lr)
        self.down.update(lr)

    def load_from_qwen3_tensors(self, gate_proj, up_proj, down_proj):
        # HF nn.Linear weights are already (out, in) == LinearLayer.W. No transpose!
        self.gate.W = gate_proj.to(self.device).contiguous()
        self.up.W = up_proj.to(self.device).contiguous()
        self.down.W = down_proj.to(self.device).contiguous()
