import torch


class Tanh:
    """
    The tanh activation function

    Attributes:
       None 
    """
    def forward(self, X):
        """
        Compute the forward pass of the tanh activation function

        Args:
            X (torch.Tensor): Input data with shape (batch_size, sequence_length, )

        Returns:
            torch.Tensor: the tanh of the input X
        """
        # TODO: Return the tanh activation
        return torch.tanh(X)

    def backward(self, dA):
        """
        Compute the backward pass of the tanh activation function

        Args:
            dA (torch.Tensor): Gradient data with shape (batch_size, sequence_length, hidden_size)

        Returns:
            torch.Tensor: dA passed into the derivative of tanh
        """
        # TODO: Return the derivative of tanh
        return 1 - dA**2

    def update(self, lr):
        """
        Update the parameters of the layer using gradient descent.

        Args:
            lr (float): Learning rate for the parameter update.

        Returns:
            None
        """
        # TODO: Update the parameters with learning rate lr
        pass
