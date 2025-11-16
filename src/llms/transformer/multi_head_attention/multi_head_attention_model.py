import torch

from src.llms.transformer.multi_head_attention.multi_head_transformer_block import MultiHeadAttentionTransformerBlock
from src.llms.transformer.one_head_attention.one_head_attention_transformer_block import \
    OneHeadAttentionTransformerBlock
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class MultiHeadAttentionLanguageModel(nn.Module):
    """
    MultiHeadAttentionLanguageModel implements a language model using multiple transformer blocks with multi-head attention.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dimension (int): Dimension of the embedding vectors.
        sequence_length (int): Maximum sequence length.
        num_blocks (int): Number of transformer blocks to instantiate.
        number_of_attention_heads (int): Number of attention heads.
        device (torch.device): Device to run the model on (CPU or CUDA).

    Returns:
        torch.Tensor: Logits for each token in the sequence.
    """

    def __init__(self,
                 vocab_size: int,  # Size of the vocabulary
                 embedding_dimension: int,  # Dimension of the token embeddings
                 sequence_length: int,  # Maximum length of input sequences
                 num_blocks: int,  # Number of transformer blocks to instantiate
                 number_of_attention_heads: int, # Number of attention heads
                 device: torch.device):  # Device to run the model on (CPU or CUDA)
        super().__init__()

        self.sequence_length = sequence_length  # Store the sequence length
        self.device = device  # Store the device

        # Token embedding layer: maps token indices to embedding vectors
        # This layer transforms each token index into a dense vector representation.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)

        # Position embedding layer: maps position indices to embedding vectors
        # This layer provides positional information to each token in the sequence,
        # allowing the model to distinguish between tokens at different positions.
        self.position_embedding = nn.Embedding(sequence_length, embedding_dimension)

        # Sequential container of transformer blocks
        # Each block applies multi-head attention and feed-forward transformations.
        # Stacking multiple blocks allows the model to learn complex dependencies.
        self.transformer_blocks = nn.Sequential(
            *[MultiHeadAttentionTransformerBlock(embedding_dimension, number_of_attention_heads) for _ in
              range(num_blocks)]
        )

        # Final layer normalization for output embeddings
        # Layer normalization stabilizes training and improves convergence.
        self.layernorm_final = nn.LayerNorm(embedding_dimension, eps=1e-5)

        # Final linear layer (language modeling head), weights tied to token embedding
        # This layer projects the final hidden states to vocabulary logits.
        # Weight tying helps regularize the model and improves generalization.
        self.language_model_head = nn.Linear(embedding_dimension, vocab_size, bias=False)
        self.language_model_head.weight = nn.Parameter(self.token_embedding.weight)

    def forward(self, input_token_indices: Tensor) -> Tensor:
        """
        Forward pass of the language model.

        Args:
            input_token_indices (Tensor): Tensor of token indices with shape [batch_size, sequence_length].

        Returns:
            Tensor: Logits for each token in the sequence, shape [batch_size, sequence_length, vocab_size].
        """
        # Get token embeddings for input tokens
        # Converts token indices into dense vectors for each token in the batch.
        token_embeddings = self.token_embedding(
            input_token_indices)  # [batch_size, sequence_length, embedding_dimension]

        # Get position embeddings for each position in the sequence
        # Generates position indices and retrieves corresponding embeddings.
        position_indices = torch.arange(input_token_indices.shape[-1], device=self.device)
        position_embeddings = self.position_embedding(position_indices)  # [sequence_length, embedding_dimension]

        # Add token and position embeddings
        # Combines semantic and positional information for each token.
        combined_embeddings = token_embeddings + position_embeddings  # [batch_size, sequence_length, embedding_dimension]

        # Pass embeddings through transformer blocks
        # Each block refines the representations using multi-head attention and feed-forward layers.
        transformer_output = self.transformer_blocks(combined_embeddings)

        # Apply final layer normalization
        # Normalizes the output to stabilize further processing.
        normalized_output = self.layernorm_final(transformer_output)

        # Project normalized output to vocabulary logits
        # Converts the final hidden states into logits for each vocabulary token.
        logits = self.language_model_head(normalized_output)  # [batch_size, sequence_length, vocab_size]

        return logits

    def generate(self,
                 input_token_indices: Tensor,
                 temperature: float = 1.0,
                 max_new_tokens: int = 50) -> Tensor:
        """
        Generate new tokens from the model given initial input.

        Args:
            input_token_indices (Tensor): Tensor of initial token indices, shape [batch_size, current_sequence_length].
            temperature (float): Sampling temperature for controlling randomness.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            Tensor: Tensor of token indices including generated tokens, shape [batch_size, current_sequence_length + max_new_tokens].
        """
        # Loop to generate new tokens one at a time
        # At each step, the model predicts the next token based on the current sequence.
        for _ in range(max_new_tokens):
            # Get logits for the last sequence window
            # Only the most recent sequence_length tokens are considered for prediction.
            logits = self(input_token_indices[:, -self.sequence_length:])  # [batch_size, sequence_length, vocab_size]

            # Select logits for the last token in the sequence
            # The last token's logits are used to sample the next token.
            last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature and softmax to get probabilities
            # Temperature controls the randomness of sampling; higher values produce more diverse outputs.
            probabilities = F.softmax(last_token_logits / temperature, dim=-1)  # [batch_size, vocab_size]

            # Sample next token from the probability distribution
            # Multinomial sampling selects the next token based on the computed probabilities.
            next_token_indices = torch.multinomial(probabilities, num_samples=1)  # [batch_size, 1]

            # Concatenate the new token to the input sequence
            # The newly generated token is appended to the sequence for the next iteration.
            input_token_indices = torch.cat((input_token_indices, next_token_indices),
                                            dim=1)  # [batch_size, current_sequence_length + 1]
        return input_token_indices
