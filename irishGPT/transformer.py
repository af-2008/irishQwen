from .linear_layer import LinearLayer
from .relu import ReLU
from .gelu import GeLU
from .layer_norm import LayerNorm
from .self_attention import SelfAttention
from .multihead_attention import MultiHeadAttention

class Transformer:
    def __init__(self, model_dim, ffn_expand=4, n_heads=1, use_gelu=False, device='cpu'):
        """
        Pre-LN transformer block with residual connections.
        model_dim: C (embedding size)
        ffn_expand: expansion factor for the FFN hidden dimension (default 4x)
        """
        self.device = device

        ffn_dim = ffn_expand * model_dim

        self.LN1 = LayerNorm(model_dim, device)
        if n_heads > 1:
            self.SA = MultiHeadAttention(model_dim, n_heads=n_heads, device=device, causal=True)
        else:
            self.SA = SelfAttention(model_dim, device=device, causal=True)
        self.LN2 = LayerNorm(model_dim, device)
        self.L1 = LinearLayer(model_dim, ffn_dim, device)
        self.R = GeLU() if use_gelu else ReLU()
        self.L2 = LinearLayer(ffn_dim, model_dim, device)

    def forward(self, X, key_pad_mask=None):
        resid = X

        X = self.LN1.forward(X)
        X = self.SA.forward(X, key_pad_mask=key_pad_mask)

        X += resid
        resid = X

        X = self.LN2.forward(X)
        X = self.L1.forward(X)
        X = self.R.forward(X)
        X = self.L2.forward(X)

        return X + resid

    def backward(self, dY):
        d_resid = dY

        dY = self.L2.backward(dY)
        dY = self.R.backward(dY)
        dY = self.L1.backward(dY)
        dY = self.LN2.backward(dY)

        dY += d_resid

        d_resid = dY

        dY = self.SA.backward(dY)
        dY = self.LN1.backward(dY)

        return dY + d_resid

    def update(self, lr):
        self.LN1.update(lr)
        self.SA.update(lr)
        self.LN2.update(lr)
        self.L1.update(lr)
        self.R.update(lr)
        self.L2.update(lr)

    def load_from_gpt2_block(self, block_state):
        self.LN1.gamma = block_state["ln_1.weight"].to(self.device).contiguous()
        self.LN1.beta = block_state["ln_1.bias"].to(self.device).contiguous()
        self.LN2.gamma = block_state["ln_2.weight"].to(self.device).contiguous()
        self.LN2.beta = block_state["ln_2.bias"].to(self.device).contiguous()

        if isinstance(self.SA, MultiHeadAttention):
            self.SA.load_from_gpt2_tensors(
                block_state["attn.c_attn.weight"],
                block_state["attn.c_attn.bias"],
                block_state["attn.c_proj.weight"],
                block_state["attn.c_proj.bias"],
            )
        else:
            raise ValueError("GPT-2 block loading requires MultiHeadAttention")

        # Our linears use X @ W.T, GPT-2 uses X @ W.
        self.L1.W = block_state["mlp.c_fc.weight"].T.to(self.device).contiguous()
        self.L1.b = block_state["mlp.c_fc.bias"].to(self.device).unsqueeze(0).contiguous()
        self.L2.W = block_state["mlp.c_proj.weight"].T.to(self.device).contiguous()
        self.L2.b = block_state["mlp.c_proj.bias"].to(self.device).unsqueeze(0).contiguous()