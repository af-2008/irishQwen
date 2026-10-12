import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .embedding import EmbeddingLayer
from .linear_layer import LinearLayer
from .transformer import Transformer
from .positional_embedding import PositionalEmbedding
from .layer_norm import LayerNorm

class IrishChat:
    def __init__(
        self,
        vocab_size: int = 50257,
        ctx_len: int = 1024,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        use_gelu: bool = True,
    ):
        self.padding_idx = 256
        self.ctx_len = ctx_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.token_embedding = EmbeddingLayer(vocab_size, d_model, device=self.device)
        self.positional_embedding = PositionalEmbedding(ctx_len, d_model, device=self.device)
        self.transformer_blocks = [
            Transformer(
                d_model,
                n_heads=n_heads,
                use_gelu=use_gelu,
                device=self.device,
            )
            for _ in range(n_layers)
        ]
        self.final_ln = LayerNorm(d_model, device=self.device)
        self.lm_head = LinearLayer(d_model, vocab_size, device=self.device)

    def load_converted_gpt2_checkpoint(self, checkpoint_path):
        """
        Load a project-native GPT-2 checkpoint (torch .pt).

        Expected top-level keys:
          - wte: (V, C)
          - wpe: (T, C)
          - blocks: list[dict] with GPT-2 block keys
          - ln_f.weight: (C,)
          - ln_f.bias: (C,)
          - optional lm_head.weight: (V, C)
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        required = ["wte", "wpe", "blocks", "ln_f.weight", "ln_f.bias"]
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

        self.token_embedding.W = ckpt["wte"].T.to(self.device).contiguous()
        self.positional_embedding.W = ckpt["wpe"].to(self.device).contiguous()
        self.final_ln.gamma = ckpt["ln_f.weight"].to(self.device).contiguous()
        self.final_ln.beta = ckpt["ln_f.bias"].to(self.device).contiguous()

        if len(ckpt["blocks"]) != len(self.transformer_blocks):
            raise ValueError(
                f"Block count mismatch: checkpoint has {len(ckpt['blocks'])}, model has {len(self.transformer_blocks)}"
            )

        for block, block_state in zip(self.transformer_blocks, ckpt["blocks"]):
            block.load_from_gpt2_block(block_state)

        if "lm_head.weight" in ckpt:
            self.lm_head.W = ckpt["lm_head.weight"].to(self.device).contiguous()
        else:
            # Standard GPT-2 uses tied embeddings.
            self.lm_head.W = self.token_embedding.W.T.contiguous()
        self.lm_head.b = torch.zeros(1, self.lm_head.W.shape[0], device=self.device)

    def forward(self, X, eval=False):
        """
        Args:
          x: (batch_size, seq_len) with integer token indices.
        Returns:
          logits: (batch_size, seq_len, vocab_size)
          outputs: (batch_size, seq_len, hidden_size) final hidden states over time.
        """
        # Padding only exists in training batches. At inference (eval=True) we never mask,
        # otherwise a generated token that happens to equal padding_idx would be hidden.
        key_pad_mask = None if eval else (X == self.padding_idx)
        B, T = X.shape

        tok = self.token_embedding.forward(X)
        pos = self.positional_embedding.forward(batch_size=B, seq_len=T)
        X = tok + pos

        for block in self.transformer_blocks:
            X = block.forward(X, key_pad_mask=key_pad_mask)

        X = self.final_ln.forward(X)
        X = self.lm_head.forward(X)

        return self.softmax(X)

    def backward(self, Y_hat, Y):
        mask = (Y.argmax(dim=-1) != self.padding_idx).unsqueeze(-1).float()
        normalizer = mask.sum().clamp_min(1.0)
        dA = (Y_hat - Y) * mask / normalizer

        dA = self.lm_head.backward(dA)
        dA = self.final_ln.backward(dA)

        for block in reversed(self.transformer_blocks):
            dA = block.backward(dA)

        self.positional_embedding.backward(dA)
        self.token_embedding.backward(dA)

    def update(self, lr):
        self.token_embedding.update(lr)
        self.positional_embedding.update(lr)
        for block in self.transformer_blocks:
            block.update(lr)
        self.final_ln.update(lr)
        self.lm_head.update(lr)

    def softmax(self, X):
        """
        Apply softmax to the input tensor.

        Args:
            X (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Tensor with softmax applied, same shape as `X`.
        """
        return F.softmax(X, dim=-1)

    def cross_entropy(self, Y_hat, Y):
        """
        Compute the cross-entropy loss.

        Args:
            Y_hat (numpy.ndarray): Predicted probability matrix of shape (n_classes, m).
            Y (numpy.ndarray): One-hot encoded true labels of shape (n_classes, m).

        Returns:
            float: The average cross-entropy loss over all m examples.

        Notes:
            A small constant epsilon is added to Y_hat to avoid computing log(0).
        """

        # TODO: Calculate the cross-entropy loss
        log_probs = torch.log(Y_hat.clamp_min(1e-12))
        token_loss = -(Y * log_probs).sum(dim=2)  # (B, T)
        mask = (Y.argmax(dim=-1) != self.padding_idx).float()
        masked_loss = token_loss * mask
        normalizer = mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def get_accuracy(self, Y_hat, Y):
        """
        Compute the classification accuracy.

        Args:
            Y_hat (numpy.ndarray): Predicted probability matrix from the network, shape (n_classes, m).
            Y (numpy.ndarray): One-hot encoded true labels, shape (n_classes, m).

        Returns:
            float: Accuracy as a fraction between 0 and 1.
        """

        # TODO: Calculate the accuracy of the network
        # labels: Y (B, C) one-hot
        Y_idx = Y.argmax(dim=-1)                # (B,)
        preds = Y_hat.argmax(dim=-1)
        mask = (Y_idx != self.padding_idx).float()
        correct = (preds == Y_idx).float() * mask
        normalizer = mask.sum().clamp_min(1.0)
        return correct.sum() / normalizer

    def chat(self, prompt_tokens, max_new_tokens=200, temperature=1.0, eos_token_id=258):
        """
        Autoregressive generation.

        Args:
            prompt_tokens: list of int token IDs (from tokenizer.encode)
            max_new_tokens: maximum number of tokens to generate
            temperature: 0 = greedy (always pick most likely),
                         >0 = sample (higher = more random)
        Returns:
            list of int token IDs (prompt + generated)
        """
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            # Truncate to context window if the sequence is too long
            x = tokens[:, -self.ctx_len:]

            # Forward pass → probabilities (B, T, V)
            probs = self.forward(x, eval=True)

            # We only care about the LAST position's prediction
            next_probs = probs[0, -1, :]        # (V,)

            # Pick the next token
            if temperature == 0:
                next_token = next_probs.argmax().item()
            else:
                # Scale probabilities by temperature and re-normalize
                scaled = next_probs.pow(1.0 / temperature)
                scaled = scaled / scaled.sum()
                next_token = torch.multinomial(scaled, 1).item()

            # Append to the sequence
            tokens = torch.cat([
                tokens,
                torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            ], dim=1)

            # Stop when EOS is produced for the active tokenizer.
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return tokens[0].tolist()

    def train(self, train_loader, epochs=100, learning_rate=0.001, verbose=True, optimizer=None):
        """
        Train the neural network using mini-batch gradient descent.

        Args:
            X (numpy.ndarray): Input data with shape (784, m), where each column is a flattened MNIST style image.
            Y (numpy.ndarray): One-hot encoded labels with shape (n_classes, m), where n_classes is 26
            epochs (int): Number of epochs for training.
            learning_rate (float): Learning rate for the parameter updates.
            batch_size (int, optional): Number of examples per mini-batch. Default is 32.
            verbose (bool, optional): If True, prints training progress every 500 epochs. Default is False.
            optimizer (optional): An optimizer from optimizers.py. If None, uses plain SGD via self.update(learning_rate).

        Returns:
            dict: A dictionary containing:
                - 'loss_history': List of loss values for each epoch.
                - 'accuracy_history': List of accuracy values for each epoch.

        Process:
            - Shuffles the dataset each epoch.
            - Processes data in mini-batches.
            - Performs a forward pass, backpropagation, and parameter updates for each mini-batch.
            - Computes the loss and accuracy for the entire dataset after each epoch.
        """
        loss_history = []
        accuracy_history = []
        
        for i in tqdm(range(epochs), desc="Epochs"):    
            batch_losses = []
            batch_accuracies = []

            for X_batch, Y_batch in tqdm(train_loader, desc="Batches"):
                # Forward propagation
                # TODO: Calculate the output of the network
                Y_hat_batch = self.forward(X_batch)

                # Calculate metrics for the whole epoch
                loss = self.cross_entropy(Y_hat_batch, Y_batch)
                accuracy = self.get_accuracy(Y_hat_batch, Y_batch)
                
                batch_losses.append(loss.item())
                batch_accuracies.append(accuracy.item())
                
                # Backward propagation
                # TODO: Calculate the gradients of the loss with respect to the input
                self.backward(Y_hat_batch, Y_batch)
                
                # Update parameters
                # TODO: Update the weights and biases of the layer using the learning rate
                if optimizer is not None:
                    optimizer.step()        # e.g. optimizers.Adam(model) -- see optimizers.py
                else:
                    self.update(learning_rate)

            loss_history.append(np.mean(batch_losses))
            accuracy_history.append(np.mean(batch_accuracies))
            
            if verbose and i % max(1, epochs // 10) == 0:
                print(f"Epoch {i+1}/{epochs}")
                print(f"loss: {loss_history[-1]:.5f}")
                print(f"accuracy: {accuracy_history[-1]:.5f}")
                print('Output test:')
                print("-" * 30)
        
        return {'loss_history': loss_history, 'accuracy_history': accuracy_history}