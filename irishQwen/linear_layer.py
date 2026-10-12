import math
import torch

class LinearLayer:
    """
    A fully connected (dense) layer that performs a linear transformation.

    Attributes:
        W (torch.Tensor): Weight matrix with shape (output_dim, input_dim).
        b (torch.Tensor): Bias vector with shape (output_dim, 1).
        X (torch.Tensor): Cached input used during the forward pass.
        dW (torch.Tensor): Gradient with respect to the weights.
        db (torch.Tensor): Gradient with respect to the biases.
    """
    def __init__(self, input_dim, output_dim, device='cpu', bias=True):
        """
        Initialize the LinearLayer with random weights and biases using He initialization.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Number of neurons (output features).
            bias (bool): If False, the layer has no bias (b is None). Qwen/Llama use no biases.

        Returns:
            None

        Notes:
            Weights and biases are initialized from a normal distribution and scaled by sqrt(2/input_dim) (He initialization).
        """
        self.device = device

        self.W = torch.randn(output_dim, input_dim, device=self.device) * math.sqrt(2.0 / input_dim)
        self.b = torch.randn(1, output_dim, device=self.device) * math.sqrt(2.0 / input_dim)

    def forward(self, X):
        """
        Compute the forward pass of the linear layer.

        Args:
            X (torch.Tensor): Input data with shape (input_dim, m) where m is the number of examples.

        Returns:
            Z (torch.Tensor): Linear output with shape (output_dim, m)

        Notes:
            The input X is stored for use during backpropagation.
        """

        # Store the input and calculate the output of the linear layer
        self.X = X

        Z = X @ self.W

        return Z += self.b



    def backward(self, dY):
        """
        Compute the backward pass of the linear layer.

        Args:
            dY (torch.Tensor): Gradient with respect to the output of the linear layer (batch_size, sequence_length, output_dim)

        Returns:
            dX (torch.Tensor): Gradient with respect to the input of the linear layer (batch_size, sequence_length, input_dim)

        Notes:
            This is trickier than in Homework 04 because our input is now 3D (batch_size, sequence_length, input_dim)
            To calculate dW we use a similar idea to im2col and flatten the input matrix across the batch_size and sequence_length dimensions.
            We then do the same thing for X and then can use a simple 2D matmul to calculate dW directly.
        """

        # dY_flat shape: (batch_size * sequence_length, output_dim)
        dY_flat = dY.reshape(-1, dY.shape[-1])

        # X_flat shape: (batch_size * sequence_length, input_dim)
        X_flat = self.X.reshape(-1, self.X.shape[-1])
        # dW shape: (output_dim, input_dim)
        self.dW = dY_flat.T @X_flat
        self.db = dY.sum(dim=(0, 1))

        dX = dY @ self.W 


        return dX



    def update(self, lr):
        """
        Update the parameters of the layer using gradient descent.

        Args:
            lr (float): Learning rate for the parameter update.

        Returns:
            None
        """

        # Update the weights and biases of the layer using the learning rate

        self.W -= lr * self.dW
        self.b -= lr * self.db

        
