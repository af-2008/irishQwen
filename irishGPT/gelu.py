import math
import torch


class GeLU:
    """
    GELU activation using the GPT-style tanh approximation.
    """

    def forward(self, X):
        self.X = X
        c = math.sqrt(2.0 / math.pi)
        self.u = c * (X + 0.044715 * X * X * X)
        self.tanh_u = torch.tanh(self.u)
        return 0.5 * X * (1.0 + self.tanh_u)

    def backward(self, dA):
        x = self.X
        c = math.sqrt(2.0 / math.pi)
        sech2 = 1.0 - self.tanh_u * self.tanh_u
        du_dx = c * (1.0 + 3.0 * 0.044715 * x * x)
        dgelu_dx = 0.5 * (1.0 + self.tanh_u) + 0.5 * x * sech2 * du_dx
        return dA * dgelu_dx

    def update(self, lr):
        # Activation layer has no trainable parameters.
        pass
