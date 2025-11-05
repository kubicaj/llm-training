import numpy as np
import requests
import matplotlib.pyplot as plt

# pytorch stuff
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from src.llms.common.llm_model import LLMModel
from src.llms.common.token_dataset import TokenDataset

tokenizer = tiktoken.get_encoding('cl100k_base')

# ///////////////////////////////////////////////////////////////////////////////
# Hyperparameters
# ///////////////////////////////////////////////////////////////////////////////

# data hyperparameters
seq_len = 8  # aka context length
stride = 2
n_vocab = tokenizer.n_vocab

# model hyperparameters
embed_dim = 2 ** 6  # 64

batch_size = 5

# ///////////////////////////////////////////////////////////////////////////////
# Get Data
# ///////////////////////////////////////////////////////////////////////////////

# tokenize the text
text = requests.get('https://www.gutenberg.org/files/35/35-0.txt').text

# text needs to be pytorch tensors
tokens = tokenizer.encode(text)
print(f'Variable "tokens" is type {type(tokens)}')

# convert to pytorch
tmTokens = torch.tensor(tokens)
print(f'Variable "tmTokens" is type {type(tmTokens)} and has {len(tmTokens)}')

# ///////////////////////////////////////////////////////////////////////////////
# Run model
# ///////////////////////////////////////////////////////////////////////////////

# new instance of the model
llm_model = LLMModel(n_vocab, embed_dim)

token_dataset = TokenDataset(tmTokens,seq_len,stride)

# get some data
X, y = token_dataset[12345]

# process the tokens (forward pass)
out = llm_model(X)

print(X.shape)
print(y.shape)
print(out.shape)  # [tokens, vocab_size]

# ///////////////////////////////////////////////////////////////////////////////
# Generate text
# ///////////////////////////////////////////////////////////////////////////////

# some text :)
generated_tokens = llm_model.generate(X.unsqueeze(dim=0), 10)

tokenizer.decode(generated_tokens[0].tolist())

# repeat multiple times from the same starting point
for i in range(5):
    # new tokens
    tokz = llm_model.generate(X.unsqueeze(dim=0), 10)
    tokz = tokz[0].tolist()

    # print our lovely poem ;)
    print(f'\n\n--- Run {i + 1} ---')
    print(tokenizer.decode(tokz))

# ///////////////////////////////////////////////////////////////////////////////
# Generate text in batches
# ///////////////////////////////////////////////////////////////////////////////

# also need a dataloader
dataloader = DataLoader(
                token_dataset,
                batch_size = batch_size,
                shuffle    = True,
                drop_last  = True
            )

# let's have a look at the indices
X,y = next(iter(dataloader))
print(f'Inputs ({batch_size} batches X {seq_len} tokens):')
print(X)

# get model outputs (logits)
out = llm_model(X)
print(out.shape) # [batch, tokens, vocab]
print('\n',out)

# generate some data
gen_tokens = llm_model.generate(X)
print(gen_tokens.shape) # [batch, (tokens+n_new_tokens)]

# repeat multiple times from the same starting point
for batchtok in gen_tokens:
  print('\n--- NEXT SAMPLE: ---\n')
  print(tokenizer.decode(batchtok.tolist()))