import torch
import math

class EmbeddingLayer():
    """
    An embedding layer 

    Attributes:
        vocab_size (int): Size of the vocabulary in tokens
        embed_dim (int): Size of the token embedding
        tokens (torch.Tensor): Cached input used during the forward pass.
        W (torch.Tensor): Weight matrix with shape (output_dim, input_dim).
        dW (torch.Tensor): Gradient with respect to the weights.
    """
    def __init__(self, vocab_size, embed_dim, device="cpu"):
        """
        Initialize the EmbeddingLayer

        Args:
            vocab_size (int): Number of tokens in the vocab
            embed_dim (int): Size of the token embedding

        Returns:
            None

        Notes:
            Weights are initialized from a normal distribution and scaled by sqrt(2/vocab_size) (He initialization).
        """

        self.device = device
        self.vocab_size = vocab_size

    def forward(self, tokens):
        """
        Compute the forward pass of the embedding layer.

        Args:
            tokens (torch.Tensor): Input data with shape (batch_size, sequence_length)

        Returns:
            torch.Tensor: Embedding output with shape (batch_size, sequence_length, embed_dim)
        """

        self.tokens = tokens

        # W[:, tokens] gives (embedding_dim, batch_size, sequence_length). 

        # Move E to last axis -> (batch_size, sequence_length, embedding_dim)

        # Return contiguous tensor for memory contiguity

    def backward(self, dY):
        """
        Compute the backward pass of the embedding layer.

        Args:
            dY (torch.Tensor): Gradient with respect to the output of the embedding layer

        Returns:
            None (Predicated on being the first layer in the model)
        """

        tokens = self.tokens
        embedding_dim, vocab_size = self.W.shape

        # accumulate columns: dW[:, tokens_flat[k]] += dY_flat[:, k]

    def update(self, lr):
        """
        Update the parameters of the layer using gradient descent.

        Args:
            lr (float): Learning rate for the parameter update.

        Returns:
            None
        """

        # Update the weights of the layer using the learning rate
        pass