import torch
import math
from collections import namedtuple
from .tanh_activation import Tanh

class MagicBox:
    """
    A Magic Box

    Attributes:
        W (torch.Tensor): Input Weight matrix with shape (hidden_size, input_dim).
        U (torch.Tensor): Hidden Weight matrix with shape (hidden_size, hidden_size).
        b (torch.Tensor): Bias vector with shape (hidden_size).
        Grad_Info (namedtuple): Cached hidden state info using the Grad_Info tuple made in forward and used in backward
        dW (torch.Tensor): Gradient with respect to the input weights.
        dU (torch.Tensor): Gradient with respect to the hidden weights.
        db (torch.Tensor): Gradient with respect to the biases.
    """

    Grad_Info = namedtuple('Grad_Info', ['x_at_timestep', 'h_at_timestep', 'h_prev_at_timestep'])

    def __init__(self, input_dim, hidden_size, device='cpu'):
        self.input_dim = input_dim      # This should match the embedding size.
        self.hidden_size = hidden_size
        self.device = device

        self.W = torch.randn(hidden_size, input_dim, device=self.device) * math.sqrt(2.0 / input_dim)
        self.U = torch.randn(hidden_size, hidden_size, device=self.device) * math.sqrt(2.0 / input_dim)
        self.b = torch.zeros(hidden_size, device=self.device)

        self.activation = Tanh()
        self.hidden_states = []

    def forward(self, X):
        """
        Compute the forward pass of the recurrent block over the entire input sequence

        Args:
            X (torch.Tensor): embedded input with shape: (batch_size, sequence_len, hidden_size)

        Returns:
            self.outputs (torch.Tensor): matrix containing the hiddent state at each timestep with shape: (batch_size, sequence_len, hidden_size)
        """
        batch_size, seq_len, _ = X.shape

        outputs = []
        self.hidden_states = [self.Grad_Info(
            x_at_timestep=None,
            h_at_timestep=torch.zeros((batch_size, self.hidden_size), device=self.device),
            h_prev_at_timestep=None
        )]

        for timestep in range(seq_len):
            x_at_timestep = X[:, timestep, :]

            # TODO: Compute the forward pass for each item in the sequence
            h_prev_at_timestep = self.hidden_states[-1].h_at_timestep
            z_at_timestep = x_at_timestep @ self.W.t() + h_prev_at_timestep @ self.U.t() + self.b  
            h_at_timestep = self.activation.forward(z_at_timestep)

            self.hidden_states.append(self.Grad_Info(
                x_at_timestep = x_at_timestep, 
                h_at_timestep = h_at_timestep,
                h_prev_at_timestep = h_prev_at_timestep))

            outputs.append(h_at_timestep)

        # Stack outputs along the time-axis: (batch_size, seq_len, hidden_size)
        self.outputs = torch.stack(outputs, dim=1)
        return self.outputs

    def backward(self, d_outputs):
        """
        Compute the backward pass of the recurrent block using backpropagation through time (BPTT)

        Args:
            d_outputs (torch.Tensor): Gradient with respect to outputs with shape: (batch_size, sequence_len, hidden_size)

        Returns:
            torch.Tensor: dX after BPTT is performed with shape: (batch_size, sequence_len, embedding_size)
        """
        # d_outputs: 
        batch_size, seq_len, hidden_size = d_outputs.shape

        dW = torch.zeros_like(self.W, device=self.device)
        dU = torch.zeros_like(self.U, device=self.device)
        db = torch.zeros_like(self.b, device=self.device)
        dX = torch.zeros((batch_size, seq_len, self.input_dim), device=self.device)

        dh_next = torch.zeros((batch_size, hidden_size), device=self.device)  # Gradient propagated from future time steps.

        for backwards_timestep in list(reversed(range(seq_len))):
            # TODO: Compute the backward pass for each item in the sequence

            dcurrent = d_outputs[:, backwards_timestep, :] + dh_next

            timestep_grad_info = self.hidden_states[backwards_timestep + 1]

            dtanh = self.activation.backward(timestep_grad_info.h_at_timestep)  # (batch_size, hidden_size)
            dz_at_timestep = dcurrent * dtanh

            # Gradients for parameters:
            dW += dz_at_timestep.T @ timestep_grad_info.x_at_timestep            # (hidden_size, input_dim)
            dU += dz_at_timestep.T @ timestep_grad_info.h_prev_at_timestep           # (hidden_size, hidden_size)
            db += dz_at_timestep.sum(dim=0)
            # Gradient with respect to input x_t.
            dX[:, backwards_timestep, :] = dz_at_timestep @ self.W
            # Propagate gradient to previous hidden state.
            dh_next = dz_at_timestep @ self.U

        self.dW = dW
        self.dU = dU
        self.db = db

        return dX

    def update(self, lr):
        """
        Update the parameters of the block using gradient descent.

        Args:
            lr (float): Learning rate for the parameter update.

        Returns:
            None
        """
        # TODO: Update the parameters with learning rate lr
        self.W -= lr * self.dW
        self.U -= lr * self.dU
        self.b -= lr * self.db
