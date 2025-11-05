import numpy as np
import requests
import matplotlib.pyplot as plt

# pytorch stuff
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F


class LLMModel(nn.Module):
    def __init__(self, n_vocab, embed_dim):
        super().__init__()

        # embedding matrix
        self.embedding = nn.Embedding(n_vocab, embed_dim)

        # unembedding (linear layer)
        self.gelu = nn.GELU()
        self.finalLinear = nn.Linear(embed_dim, n_vocab, bias=False)

    def forward(self, tokx):
        # forward pass
        x = self.embedding(tokx)  # [batch, token, embed_dim]
        x = self.gelu(x)
        x = self.finalLinear(x)  # [batch, token, vocab_size]

        # note: no softmax here!
        return x  # logits

    def generate(self, tokx, n_new_tokens=30):
        # tokx is [batch, tokens]

        for _ in range(n_new_tokens):
            # get predictions
            x = self(tokx)

            # extract the final token to predict the next
            x = x[:, -1, :]  # [batch, vocab_size]

            # apply softmax to get probability values over all tokens in the vocab
            probs = F.softmax(x, dim=-1)

            # probabilistically sample from the distribution
            tokx_next = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # append
            tokx = torch.cat((tokx, tokx_next), dim=1)  # [batch, (tokens+1)]
        return tokx
