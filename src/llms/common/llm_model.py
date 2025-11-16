import numpy as np
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
        super().__init__()  # Initialize base nn.Module

        # nn.Embedding: Maps token IDs (integers) to learned dense vectors.
        # Shape: (n_vocab, embed_dim)
        self.embedding = nn.Embedding(n_vocab, embed_dim)

        # nn.Embedding: Maps position indices (0..seq_len-1) to learned vectors.
        # Shape: (seq_len, embed_dim)
        self.positions = nn.Embedding(seq_len, embed_dim)

        # GELU activation: Smooth nonlinearity, common in transformers.
        self.gelu = nn.GELU()

        # LayerNorm: Normalizes embeddings for stability and better training.
        # Operates over last dimension (embed_dim).
        self.layernorm = nn.LayerNorm(embed_dim)

        # Linear layer: Projects embeddings to vocabulary logits.
        # Shape: (embed_dim, n_vocab)
        # No bias for weight tying.
        self.finalLinear = nn.Linear(embed_dim, n_vocab, bias=False)

        # Weight tying: Share weights between input embedding and output layer.
        # This reduces parameters and improves generalization.
        self.finalLinear.weight = nn.Parameter(self.embedding.weight)

        # Store embedding dimension and sequence length for later use.
        self.embed_dim = embed_dim
        self.seq_len = seq_len

    def forward(self, tokx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict next-token logits.

        Parameters:
            tokx (torch.Tensor): A batch of token sequences (batch_size x seq_len)
                - Each entry is an integer token ID.

        Returns:
            torch.Tensor: Predicted logits for each token in the vocabulary.
                          Shape = (batch_size, seq_len, n_vocab)
        """

        # Step 1: Convert token IDs to learned word vectors.
        # token_embed: (batch_size, seq_len, embed_dim)
        token_embed = self.embedding(tokx)

        # Step 2: Create position indices [0, 1, ..., seq_len-1] for each token position.
        # posit_embed: (seq_len, embed_dim)
        posit_embed = self.positions(torch.arange(tokx.shape[-1]))

        # Step 3: Add token embeddings and positional embeddings.
        # Broadcasting posit_embed across batch dimension.
        # x: (batch_size, seq_len, embed_dim)
        x = token_embed + posit_embed

        # Step 4: Normalize embeddings for stability.
        # x: (batch_size, seq_len, embed_dim)
        x = self.layernorm(x)

        # Step 5: Apply GELU non-linearity.
        # x: (batch_size, seq_len, embed_dim)
        x = self.gelu(x)

        # Step 6: Project embeddings to vocabulary logits.
        # x: (batch_size, seq_len, n_vocab)
        # Divide by sqrt(embed_dim) for scaling (helps training stability).
        x = self.finalLinear(x) / np.sqrt(self.embed_dim)

        # Step 7: Return logits (unnormalized scores for each vocab token).
        return x

    def generate(self, tokx: torch.Tensor, temperature: float = 1.0, n_new_tokens: int = 50) -> torch.Tensor:
        """
        Generate new tokens, one at a time, using sampling.

        Parameters:
            tokx (torch.Tensor): Starting token sequence (batch_size x sequence_length)
                - Each entry is an integer token ID.
            temperature (float): Controls randomness; lower = more confident, higher = more random.
            n_new_tokens (int): How many tokens to generate.

        Returns:
            torch.Tensor: The input sequence extended with generated tokens.
        """

        # Repeat token generation step n_new_tokens times.
        for _ in range(n_new_tokens):
            # Only keep the latest seq_len tokens (model’s max memory window).
            # x: (batch_size, seq_len, n_vocab)
            x = self(tokx[:, -self.seq_len:])

            # Take the model's prediction for the last token in the sequence.
            # x: (batch_size, n_vocab)
            x = x[:, -1, :]

            # Convert raw logits to probability values using softmax.
            # Divide by temperature to control randomness.
            # probs: (batch_size, n_vocab)
            probs = F.softmax(x / temperature, dim=-1)

            # Randomly sample the next token according to the probability distribution.
            # tokx_next: (batch_size, 1)
            tokx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to the sequence.
            # tokx: (batch_size, sequence_length + 1)
            tokx = torch.cat((tokx, tokx_next), dim=1)

        # Return the full sequence (original + generated tokens).
        return tokx
