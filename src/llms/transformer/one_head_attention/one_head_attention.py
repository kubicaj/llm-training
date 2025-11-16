import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class OneHeadAttention(nn.Module):
    """
    Implements a single attention head for a transformer model.

    Args:
        embedding_dimension (int): The size of the input and output embeddings for the attention head.

    Attributes:
        key_projection (nn.Linear): Linear layer to project input embeddings to key vectors.
        query_projection (nn.Linear): Linear layer to project input embeddings to query vectors.
        value_projection (nn.Linear): Linear layer to project input embeddings to value vectors.
        output_projection (nn.Linear): Linear layer applied after attention to mix output features.
    """

    def __init__(self, embedding_dimension: int):
        """
        Initializes the OneAttentionHead module.

        Args:
            embedding_dimension (int): Dimensionality of the input embeddings.
        Returns:
            None
        """
        super().__init__()  # Call the parent class constructor

        # Linear layer to project input embeddings to key vectors.
        # Input shape: (batch_size, sequence_length, embedding_dimension)
        # Output shape: (batch_size, sequence_length, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Linear layer to project input embeddings to query vectors.
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Linear layer to project input embeddings to value vectors.
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Linear layer to mix output features after attention.
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

    def forward(self, input_embeddings: Tensor) -> Tensor:
        """
        Forward pass for the attention head.

        Args:
            input_embeddings (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).
                Represents the sequence of token embeddings for a batch.

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dimension).
                Represents the attended and projected token embeddings.
        """

        # Project input embeddings to key vectors.
        key_vectors = self.key_projection(input_embeddings)  # Shape: (batch_size, sequence_length, embedding_dimension)

        # Project input embeddings to query vectors.
        query_vectors = self.query_projection(
            input_embeddings)  # Shape: (batch_size, sequence_length, embedding_dimension)

        # Project input embeddings to value vectors.
        value_vectors = self.value_projection(
            input_embeddings)  # Shape: (batch_size, sequence_length, embedding_dimension)

        # Compute scaled dot-product attention.
        # query_vectors: queries for each token
        # key_vectors: keys for each token
        # value_vectors: values for each token
        # is_causal=True: ensures each position only attends to previous positions (for autoregressive models)
        attention_output = F.scaled_dot_product_attention(
            query_vectors, key_vectors, value_vectors, is_causal=True
        )  # Shape: (batch_size, sequence_length, embedding_dimension)

        # Apply final linear transformation to the attention output.
        projected_output = self.output_projection(
            attention_output)  # Shape: (batch_size, sequence_length, embedding_dimension)

        # Return the output tensor.
        return projected_output
