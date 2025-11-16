import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention implements the multi-head self-attention mechanism used in transformer models.
    This mechanism allows the model to jointly attend to information from different representation subspaces
    at different positions, which is crucial for capturing complex dependencies in sequences.

    Args:
        embedding_dimension (int): The dimensionality of the input embeddings.
        number_of_attention_heads (int): The number of parallel attention heads.

    Attributes:
        num_heads (int): Stores the number of attention heads.
        head_dim (int): Stores the dimensionality of each attention head.
        QKV (nn.Linear): Linear layer to compute concatenated Query, Key, and Value matrices.
        W0 (nn.Linear): Linear layer to mix the outputs from all attention heads.
    """

    def __init__(self, embedding_dimension: int, number_of_attention_heads: int):
        """
        Initializes the MultiHeadAttention module.

        Args:
            embedding_dimension (int): The dimensionality of the input embeddings.
            number_of_attention_heads (int): The number of parallel attention heads.

        Returns:
            None
        """
        super().__init__()

        # Store the number of attention heads for use in forward computations.
        self.num_heads = number_of_attention_heads

        # Calculate the dimension of each attention head by dividing the embedding dimension by the number of heads.
        # This ensures that the total output dimension remains consistent with the input embedding dimension.
        self.head_dim = embedding_dimension // number_of_attention_heads

        # Linear layer to project the input tensor into concatenated Query, Key, and Value matrices.
        # The output dimension is three times the embedding dimension to accommodate Q, K, and V.
        self.QKV = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=True)

        # Linear layer to mix the outputs from all attention heads back into the original embedding dimension.
        # This step is essential for combining the information from different heads.
        self.W0 = nn.Linear(embedding_dimension, embedding_dimension, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Performs the forward pass of multi-head attention.

        Args:
            input_tensor (Tensor): Input tensor of shape [batch_size, sequence_length, embedding_dimension].
                - batch_size (int): Number of samples in the batch.
                - sequence_length (int): Length of the input sequence.
                - embedding_dimension (int): Dimensionality of each input embedding.

        Returns:
            Tensor: Output tensor of shape [batch_size, sequence_length, embedding_dimension].
        """

        # Extract batch size, sequence length, and embedding dimension from the input tensor's shape.
        # This information is used to reshape and process the tensor throughout the attention mechanism.
        batch_size, sequence_length, embedding_dimension = input_tensor.shape  # [batch_size, sequence_length, embedding_dimension]

        # Project the input tensor into concatenated Query, Key, and Value matrices using the QKV linear layer.
        # This step prepares the data for the attention computation by generating the necessary components.
        qkv_concat = self.QKV(input_tensor)  # [batch_size, sequence_length, 3 * embedding_dimension]

        # Split the concatenated QKV tensor into separate Query, Key, and Value matrices along the last dimension.
        # Each matrix will have the shape [batch_size, sequence_length, embedding_dimension].
        query_matrix, key_matrix, value_matrix = torch.split(
            qkv_concat, embedding_dimension, dim=2
        )  # Each: [batch_size, sequence_length, embedding_dimension]

        # Reshape the Query, Key, and Value matrices to expose the head dimension.
        # The new shape is [batch_size, sequence_length, num_heads, head_dim].
        # Transpose to [batch_size, num_heads, sequence_length, head_dim] to facilitate parallel attention computation.
        query_matrix = query_matrix.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_matrix = key_matrix.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_matrix = value_matrix.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        # This transformation enables each head to process its own slice of the input independently.

        # Apply scaled dot-product attention for each head.
        # The function computes attention scores, applies causal masking (for autoregressive models), and combines values.
        # is_causal=True ensures that each position can only attend to previous positions, preserving the autoregressive property.
        attention_output = F.scaled_dot_product_attention(
            query_matrix, key_matrix, value_matrix, is_causal=True
        )  # [batch_size, num_heads, sequence_length, head_dim]

        # Transpose the attention output back and reshape to combine all heads into a single embedding dimension.
        # The resulting tensor has the shape [batch_size, sequence_length, embedding_dimension].
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, embedding_dimension)
        # This step aggregates the information from all heads for each position in the sequence.

        # Apply the final linear layer to mix the outputs from all attention heads.
        # This produces the final output tensor, ready for downstream processing in the transformer.
        output_tensor = self.W0(attention_output)  # [batch_size, sequence_length, embedding_dimension]

        # Return the output tensor containing the attended representations.
        return output_tensor

