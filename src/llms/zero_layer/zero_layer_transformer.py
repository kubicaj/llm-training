import torch


class ZeroLayerTransformer(torch.nn.Module):

    def __init__(self, vocab_size: int, model_dimension: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.embed = torch.nn.Embedding(vocab_size, model_dimension)
        self.unembed = torch.nn.Linear(model_dimension, vocab_size, bias=False)

    def forward(self, token_ids):
        embeddings = self.embed(token_ids)
        return self.unembed(embeddings)