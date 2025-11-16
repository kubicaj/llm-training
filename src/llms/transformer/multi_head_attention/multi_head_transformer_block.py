import torch.nn as nn
from torch import Tensor

from src.llms.transformer.multi_head_attention.multi_head_attention import MultiHeadAttention


class MultiHeadAttentionTransformerBlock(nn.Module):
    """
    MultiHeadAttentionTransformerBlock applies a multi-head attention mechanism followed by a feedforward
    neural network (MLP).

    Args:
        embedding_dimension (int): The size of the input and output embeddings.
        number_of_attention_heads (int): The number of attention heads to use in the multi-head attention mechanism.
    """
    def __init__(self, embedding_dimension: int, number_of_attention_heads: int):
        super().__init__()

        # Layer normalization before attention subblock
        self.layernorm_1 = nn.LayerNorm(embedding_dimension, eps=1e-5)
        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(embedding_dimension, number_of_attention_heads)

        # Layer normalization before feedforward subblock
        self.layernorm_2 = nn.LayerNorm(embedding_dimension, eps=1e-5)
        # First linear layer in feedforward subblock, expands embedding size by 4x
        self.feedforward_linear_1 = nn.Linear(embedding_dimension, 4 * embedding_dimension, bias=True)
        # GELU activation function for non-linearity
        self.gelu_activation = nn.GELU()
        # Second linear layer in feedforward subblock, contracts back to embedding size
        self.feedforward_linear_2 = nn.Linear(4 * embedding_dimension, embedding_dimension, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass for the transformer block.

        Args:
            input_tensor (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor: Output tensor of the same shape as input_tensor.
        """

        # Apply layer normalization before attention
        normalized_tensor_for_attention = self.layernorm_1(input_tensor)
        # Apply multi-head attention and add residual connection
        attention_output = input_tensor + self.attention(normalized_tensor_for_attention)

        # Apply layer normalization before feedforward subblock
        normalized_tensor_for_feedforward = self.layernorm_2(attention_output)
        # Apply feedforward network: expand, activate, contract, and add residual connection
        feedforward_output = attention_output + self.feedforward_linear_2(
            self.gelu_activation(
                self.feedforward_linear_1(normalized_tensor_for_feedforward)
            )
        )

        # Return the final output tensor
        return feedforward_output
