import torch

class ReLU:
    """
    Element-wise rectified linear activation.
    """
    def forward(self, X):
        """
        Apply ReLU activation and cache the input tensor.

        Args:
            X (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Tensor with negatives zeroed out, same shape as `X`.
        """

        # TODO: Store the input and calculate and return the output of the ReLU layer
        self.X = X
        return torch.maximum(torch.tensor(0.0, device=X.device), X)

    def backward(self, dA):
        """
        Propagate gradients through the ReLU non-linearity.

        Args:
            dA (torch.Tensor): Upstream gradient matching the shape of the forward output.

        Returns:
            torch.Tensor: Gradient with respect to the input, zeroed where the cached input was non-positive.
        """

        # TODO: Calculate and return the gradient of the loss with respect to the input
        return dA * (self.X > 0)

    def update(self, lr):
        """
        Keep API parity with trainable layers; ReLU has no parameters to update.

        Args:
            lr (float): Unused learning rate argument.

        Returns:
            None
        """
        # TODO: Update the weights and biases of the layer using the learning rate
        pass
