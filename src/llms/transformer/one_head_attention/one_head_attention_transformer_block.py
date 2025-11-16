import torch.nn as nn
from torch import Tensor

from src.llms.transformer.one_head_attention.one_head_attention import OneHeadAttention


class OneHeadAttentionTransformerBlock(nn.Module):
    """
    TransformerBlock implements a single block of a transformer architecture.
    It consists of an attention sublayer and a feedforward (MLP) sublayer.
    """

    def __init__(self, embedding_dimension: int):
        """
        Initializes the TransformerBlock.

        Args:
            embedding_dimension (int): The size of the input and output embeddings.
        """
        super().__init__()

        # Layer normalization before attention sublayer
        self.layer_norm_attention = nn.LayerNorm(embedding_dimension)
        # Single attention head operating on normalized input
        self.attention_head = OneHeadAttention(embedding_dimension)

        # Layer normalization before MLP sublayer
        self.layer_norm_mlp = nn.LayerNorm(embedding_dimension)
        # First linear layer expands embedding dimension by 4x
        self.linear_expand = nn.Linear(embedding_dimension, 4 * embedding_dimension)
        # GELU nonlinearity for MLP
        self.gelu_activation = nn.GELU()
        # Second linear layer contracts back to original embedding dimension
        self.linear_contract = nn.Linear(4 * embedding_dimension, embedding_dimension)

    def forward(self, input_embeddings: Tensor) -> Tensor:
        """
        Forward pass for the TransformerBlock.

        Args:
            input_embeddings (Tensor):
                Input tensor of shape (batch_size, sequence_length, embedding_dimension).
                Represents the input embeddings for the block.

        Returns:
            Tensor:
                Output tensor of the same shape as input_embeddings.
                Represents the transformed embeddings after attention and MLP sublayers.
        """

        # -------- attention sublayer --------
        # Apply layer normalization to input embeddings
        normalized_embeddings = self.layer_norm_attention(input_embeddings)
        # Pass normalized embeddings through the attention head
        attention_output = self.attention_head(normalized_embeddings)
        # Add residual connection (input + attention output)
        residual_attention_output = input_embeddings + attention_output
        # ------------------------------------

        # -------- MLP sublayer --------
        # Apply layer normalization to the output of the attention sublayer
        normalized_attention_output = self.layer_norm_mlp(residual_attention_output)
        # Expand the embedding dimension by 4x using a linear layer
        expanded_embeddings = self.linear_expand(normalized_attention_output)
        # Apply GELU activation for non-linearity
        activated_embeddings = self.gelu_activation(expanded_embeddings)
        # Contract the embedding dimension back to original size
        contracted_embeddings = self.linear_contract(activated_embeddings)
        # Add residual connection (attention output + contracted embeddings)
        output_embeddings = residual_attention_output + contracted_embeddings
        # ------------------------------------

        # Return the final output embeddings
        return output_embeddings
