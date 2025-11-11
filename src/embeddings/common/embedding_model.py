import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    """
      A simple embedding model with an embedding layer followed by two linear layers.
      Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding vectors.
        context_size (int): Number of words in the context.
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(EmbeddingModel, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # linear layers
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        loop all layers linear1 and linear2 to get result

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, context_size).

        Return:
            log_probs (torch.Tensor): Log probabilities of shape (batch_size, vocab_size).
        """

        # extract and flatten embeddings [batch_size, context_size * embedding_dim]
        embeds = self.embeddings(inputs).view(inputs.shape[0], -1)

        # fully connected layers
        linear1_out = self.linear1(embeds)
        out = F.relu(linear1_out)
        out = self.linear2(out)

        # log softmax for classification (note: NLLLoss expects logprobs as inputs)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
