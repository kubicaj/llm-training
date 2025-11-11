import numpy as np
import requests
import matplotlib.pyplot as plt

# PyTorch imports for model building and training utilities
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F


class LLMModel(nn.Module):
    """
    A very small language model using:
    - Token embeddings (convert token IDs to vectors)
    - Positional embeddings (tell the model where each token is in the sentence)
    - Layer normalization + GELU activation
    - A final linear layer tied with token embeddings for prediction
    """

    def __init__(self, n_vocab: int, embed_dim: int, seq_len: int):
        """
        Parameters:
            n_vocab (int): Number of possible tokens in the vocabulary.
            embed_dim (int): Size of each embedding vector.
            seq_len (int): Maximum number of tokens the model can see at once.
        """
        super().__init__()

        # Turns token IDs into learned continuous vectors
        self.embedding = nn.Embedding(n_vocab, embed_dim)

        # Learns a unique vector for each possible position in the sequence
        self.positions = nn.Embedding(seq_len, embed_dim)

        # Nonlinear activation used commonly in transformers
        self.gelu = nn.GELU()

        # Normalizes embedding values so they stay numerically stable
        self.layernorm = nn.LayerNorm(embed_dim)

        # Final prediction layer.
        # It maps embeddings back to vocabulary size to guess the next token.
        self.finalLinear = nn.Linear(embed_dim, n_vocab, bias=False)

        # "Weight tying": share weights between input embeddings and output layer.
        # This improves model quality and saves memory.
        self.finalLinear.weight = nn.Parameter(self.embedding.weight)

        # Store useful sizes
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def forward(self, tokx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict next-token logits.

        Parameters:
            tokx (torch.Tensor): A batch of token sequences (batch_size x seq_len)

        Returns:
            torch.Tensor: Predicted logits for each token in the vocabulary.
                          Shape = (batch_size, seq_len, n_vocab)
        """

        # Convert token IDs to learned word vectors
        token_embed = self.embedding(tokx)

        # Create a tensor of position indices [0, 1, 2, ..., seq_len-1]
        # and convert to position embeddings
        posit_embed = self.positions(torch.arange(tokx.shape[-1]))

        # Add token meaning vectors + positional location vectors
        # Broadcasting automatically adds the positions across the batch
        x = token_embed + posit_embed

        # Normalize values to keep them stable
        x = self.layernorm(x)

        # Apply GELU non-linearity
        x = self.gelu(x)

        # Convert embedding vectors into logits over vocabulary
        # Divide by sqrt(embed_dim) to scale values (helps training stability)
        x = self.finalLinear(x) / np.sqrt(self.embed_dim)

        return x

    def generate(self, tokx: torch.Tensor, temperature: float = 1.0, n_new_tokens: int = 50) -> torch.Tensor:
        """
        Generate new tokens, one at a time, using sampling.

        Parameters:
            tokx (torch.Tensor): Starting token sequence (batch_size x sequence_length)
            temperature (float): Controls randomness; lower = more confident, higher = more random
            n_new_tokens (int): How many tokens to generate

        Returns:
            torch.Tensor: The input sequence extended with generated tokens
        """

        # Repeat token generation step n_new_tokens times
        for _ in range(n_new_tokens):

            # Only keep the latest seq_len tokens (model’s max memory window)
            x = self(tokx[:, -self.seq_len:])  # Shape: (batch, seq_len, vocab)

            # Take the model's prediction for the last token in the sequence
            x = x[:, -1, :]  # Shape: (batch, vocab)

            # Convert raw logits to probability values using softmax
            # Divide by temperature to control randomness
            probs = F.softmax(x / temperature, dim=-1)

            # Randomly sample the next token according to the probability distribution
            tokx_next = torch.multinomial(probs, num_samples=1)  # Shape: (batch, 1)

            # Append sampled token to the sequence
            tokx = torch.cat((tokx, tokx_next), dim=1)

        return tokx
