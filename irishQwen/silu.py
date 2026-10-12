import torch


class SiLU:
    """
    SiLU / "Swish" activation: x * sigmoid(x).
    This is the activation inside SwiGLU (Llama, Qwen, Mistral, ...).
    """

    def forward(self, X):
        self.X = X
        self.sig = torch.sigmoid(X)
        return X * self.sig

    def backward(self, dA):
        # d/dx [x * s(x)] = s(x) + x * s(x) * (1 - s(x))
        x = self.X
        dsilu_dx = self.sig * (1.0 + x * (1.0 - self.sig))
        return dA * dsilu_dx

    def update(self, lr):
        # Activation layer has no trainable parameters.
        pass
