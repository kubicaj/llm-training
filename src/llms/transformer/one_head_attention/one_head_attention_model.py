import torch

from src.llms.transformer.one_head_attention.one_head_attention_transformer_block import OneHeadAttentionTransformerBlock
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class OneHeadAttentionLanguageModel(nn.Module):
    """
    OneAttentionLanguageModel implements a language model using multiple transformer blocks
    with a single attention mechanism.

    Args:
        num_transformer_blocks (int): Number of transformer blocks to use in the model.
        vocab_size (int): Size of the vocabulary.
        embedding_dimension (int): Dimensionality of the token embeddings.
        sequence_length (int): Maximum sequence length for positional embeddings.

    Returns:
        torch.Tensor: Output logits of shape [batch_size, sequence_length, vocab_size] in forward().
        torch.Tensor: Generated token indices of shape [batch_size, sequence_length + n_new_tokens] in generate().
    """

    def __init__(
            self,
            num_transformer_blocks: int,
            vocab_size: int,
            embedding_dimension: int,
            sequence_length: int
    ):
        super().__init__()
        # Store the maximum sequence length for positional embeddings
        self.sequence_length = sequence_length

        # Token embedding matrix: maps token indices to embedding vectors
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        # Positional embedding matrix: maps position indices to embedding vectors
        self.position_embedding = nn.Embedding(sequence_length, embedding_dimension)

        # Stack of transformer blocks, each with single attention mechanism
        self.transformer_blocks = nn.Sequential(
            *[OneHeadAttentionTransformerBlock(embedding_dimension) for _ in range(num_transformer_blocks)]
        )

        # Final layer normalization after transformer blocks
        self.final_layer_norm = nn.LayerNorm(embedding_dimension)
        # Linear layer to project embeddings to vocabulary logits
        self.output_linear = nn.Linear(embedding_dimension, vocab_size, bias=False)

        # Tie the output linear layer's weights to the token embedding weights
        self.output_linear.weight = nn.Parameter(self.token_embedding.weight)

    def forward(self, token_indices: Tensor) -> Tensor:
        """
        Forward pass of the language model.

        Args:
            token_indices (Tensor): Input tensor of token indices with shape [batch_size, sequence_length].

        Returns:
            Tensor: Output logits of shape [batch_size, sequence_length, vocab_size].
        """
        # Get token embeddings for input token indices
        token_embeddings = self.token_embedding(token_indices)  # [batch_size, sequence_length, embedding_dimension]

        # Get positional embeddings for each position in the sequence
        position_indices = torch.arange(token_indices.shape[-1], device=token_indices.device)
        position_embeddings = self.position_embedding(position_indices)  # [sequence_length, embedding_dimension]

        # Add token and positional embeddings
        input_embeddings = token_embeddings + position_embeddings  # [batch_size, sequence_length, embedding_dimension]

        # Pass through stacked transformer blocks
        transformer_output = self.transformer_blocks(input_embeddings)

        # Apply final layer normalization
        normalized_output = self.final_layer_norm(transformer_output)

        # Project to vocabulary logits
        output_logits = self.output_linear(normalized_output)

        # Return logits for each token position
        return output_logits

    def generate(
            self,
            input_token_indices: Tensor,
            temperature: float = 1.0,
            num_new_tokens: int = 50
    ) -> Tensor:
        """
        Generate new tokens autoregressively from the model.

        Args:
            input_token_indices (Tensor): Initial input tensor of token indices [batch_size, initial_sequence_length].
            temperature (float): Sampling temperature for softmax distribution.
            num_new_tokens (int): Number of new tokens to generate.

        Returns:
            Tensor: Tensor of token indices including generated tokens
            [batch_size, initial_sequence_length + num_new_tokens].
        """
        for _ in range(num_new_tokens):
            # Only use the last 'sequence_length' tokens for prediction
            current_input = input_token_indices[:, -self.sequence_length:]

            # Get output logits from the model
            output_logits = self(current_input)  # [batch_size, sequence_length, vocab_size]

            # Select logits for the last token in the sequence
            last_token_logits = output_logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply softmax with temperature to get probabilities
            token_probabilities = F.softmax(last_token_logits / temperature, dim=-1)  # [batch_size, vocab_size]

            # Sample next token indices from the probability distribution
            next_token_indices = torch.multinomial(token_probabilities, num_samples=1)  # [batch_size, 1]

            # Concatenate the new token to the input sequence
            input_token_indices = torch.cat((input_token_indices, next_token_indices),
                                            dim=1)  # [batch_size, sequence_length + 1]

        # Return the full sequence including generated tokens
        return input_token_indices
