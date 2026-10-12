import torch


class SiLU:
    """
    SiLU / "Swish" activation: x * sigmoid(x).
    This is the activation inside SwiGLU (Llama, Qwen, Mistral, ...).
    """

    def forward(self, X):
        # x * sigmoid(x)
        pass

    def backward(self, dA):
        # d/dx [x * s(x)] = s(x) + x * s(x) * (1 - s(x))
        pass

    def update(self, lr):
        # Activation layer has no trainable parameters.
        pass
