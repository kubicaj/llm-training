# typical libraries...
import numpy as np

# for importing and working with texts
import requests
import re
import string

from torch import Tensor
# pytorch stuff
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from torchinfo import summary

import matplotlib_inline.backend_inline

from src.embeddings.common.embedding_model import EmbeddingModel
from src.embeddings.common import train_the_model
from src.embeddings.common.word_dataset import WordDataset

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# get raw text from internet (The Time Machine... yeah I use it a lot :P  )
text = requests.get('https://www.gutenberg.org/files/35/35-0.txt').text
# character strings to replace with space
strings2replace = [ '\r\n\r\nâ\x80\x9c','â\x80\x9c','â\x80\x9d','\r\n','â\x80\x94','â\x80\x99','â\x80\x98','_', ]

# use regular expression (re) to replace those strings with space
for str2match in strings2replace:
  text = re.compile(r'%s'%str2match).sub(' ',text)

# remove non-ASCII characters and numbers, and make lower-case
text = re.sub(r'[^\x00-\x7F]+', ' ', text)
text = re.sub(r'\d+','',text).lower()

# split into words with >1 letter
words = re.split(f'[{string.punctuation}\s]+',text)
words = [item.strip() for item in words if item.strip()]
words = [item for item in words if len(item)>1]

# create the vocabulary (lexicon)
vocab  = sorted(set(words))
nWords = len(words)
nVocab = len(vocab)

# encoder/decoder look-up-tables (as python dictionaries)
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for i,w in enumerate(vocab)}

# show a few keys in the dictionary
print(f'The book contains {nWords:,} words, {nVocab:,} of which are unique and comprise the vocab.')
print(f'\n\nFirst 10 vocab words:\n',list(word2idx.keys())[:10])

# parameters for dataset
context_length = 8 # context length
stride = 2 # skipping

# initialize
inputs  = []
targets = []

# overlapping sequences of context_length
for i in range(0,nWords-context_length,stride):

  # get a few words
  in_seq   = words[i  : i+context_length  ]
  targ_seq = words[i+1: i+context_length+1]

  # append to the lists
  inputs.append([word2idx[w] for w in in_seq])
  targets.append([word2idx[w] for w in targ_seq])

print(inputs[123])
print(targets[123])

# a closer look:
print('Inputs: ',inputs[4])
print('Targets:',targets[4])
print('')
print('Inputs :',inputs[5])
print('Targets:',targets[5])
# this is what we need, although we need it in torch Dataset/DataLoader format

# we need each list to be a tensor
tensor: Tensor = torch.tensor(inputs[4])

# create an instance!
context_length = 6 # context length
stride = 3 # skipping over tokens
text_dataset = WordDataset(words,word2idx,context_length,stride)

print(text_dataset[4])

# ///////////////////////////////////////////////////////////////////////////////
# And a dataloader for training
# ///////////////////////////////////////////////////////////////////////////////

# also need a dataloader
dataloader = DataLoader(
                text_dataset,
                batch_size = 32, # 2 for looking; 32 for training
                shuffle    = True,
                drop_last  = True
            )

# let's have a look at the indices
X,y = next(iter(dataloader))
print('Inputs:')
print(X), print('')

print('Targets:')
print(y), print('\n\n\n')

# and the words
print('Inputs in words (first batch):')
print([idx2word[item.item()] for item in X[0]])
print('')

print('Targets in words (first batch):')
print([idx2word[item.item()] for item in y[0]])

# ///////////////////////////////////////////////////////////////////////////////
# Code below is for video "Build a model to learn the embeddings"
# ///////////////////////////////////////////////////////////////////////////////

# exploring dimensionality based on vocab sizes

# vocab sizes
N = np.logspace(np.log10(1000),np.log10(100000),23)

# heuristic for non-LLM models like word2vec or glove:
embdim = np.sqrt(N)

# parameters for GPT2
gpt2dims = [ 50257,768 ]


# ///////////////////////////////////////////////////////////////////////////////
# Create and explore an embedding layer
# ///////////////////////////////////////////////////////////////////////////////

# dimensionality of embedding space (arbitrarily set to 100)
embeddingDimension = 100

# create a random embedding
embedding_layer = nn.Embedding(nVocab,embeddingDimension)

# let's see its size
shape_weight = embedding_layer.weight.shape


# embeddings for closely related words
word1 = 'time'
word2 = 'machine'

# their embeddings
embed1 = embedding_layer.weight.detach()[word2idx[word1],:]
embed2 = embedding_layer.weight.detach()[word2idx[word2],:]

# cosine similiarity between them
cosSim = torch.dot(embed1,embed2)/(torch.norm(embed1)*torch.norm(embed2))


# ///////////////////////////////////////////////////////////////////////////////
# Build the model
# ///////////////////////////////////////////////////////////////////////////////

# create a model instance!
model = EmbeddingModel(vocab_size=nVocab, embedding_dim=embeddingDimension, context_size=context_length)
print(model)

# apply Xavier weight distribution
for param in model.parameters():
  if param.dim()>1: # also excludes biases
    nn.init.xavier_normal_(param)

# let's test the model

X,y = next(iter(dataloader))
# This invokes the forward method of your model
modelOut = model(X)

print('Input to model:')
print(X), print('')

print(f'Output from model (size: {list(modelOut.detach().shape)}):')
print(modelOut)
#%%
# log soft-max output:
print(modelOut.detach()[0])
print('')

# shouldn't the sum be 1?
print(f'Log softmax sum = {modelOut.detach()[0].sum():.3f}')

# ah, it's *log* softmax :D
print(f'exp(log(softmax)) sum = {torch.exp(modelOut.detach()[0]).sum():.3f}')
#%%
# find the word with the highest probability
print('Model input:')
print([idx2word[w.item()] for w in X[0]])
print('')

print('Model output:')
print(idx2word[modelOut[0].argmax().item()])
#%%

# ///////////////////////////////////////////////////////////////////////////////
## Have the model generate text
# ///////////////////////////////////////////////////////////////////////////////

# grab some data from the loader
X,y = next(iter(dataloader))

print('First input:')
print(' '.join([idx2word[w.item()] for w in X[0]]))
print('\nSubsequent inputs:')

# text generation
for _ in range(context_length):

  # get output for this input
  Y = model(X)

  # pick the most likely next word
  nextWord = Y[0].argmax().item()

  # create new input for the next iteration (word)
  X[0] = torch.concatenate((X[0][1:],torch.tensor([nextWord])))

  # print out the generated text so far
  print(' '.join([idx2word[w.item()] for w in X[0]]))

# summary of model and parameters
summary(model, input_data=X, col_names=['input_size','output_size','num_params'])

# ///////////////////////////////////////////////////////////////////////////////
# Code below is for video "Train and evaluate the model"
# ///////////////////////////////////////////////////////////////////////////////

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# create a fresh model instance
model = EmbeddingModel(vocab_size=nVocab, embedding_dim=embeddingDimension, context_size=context_length)

# with Xavier weight distribution
for param in model.parameters():
  if param.dim()>1: nn.init.xavier_normal_(param)


# and move it to the GPU
model = model.to(device)

# create the loss and optimizer functions
loss_function = nn.NLLLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)

# quick test for errors and sanity-check the output matrix sizes
X,y = next(iter(dataloader))
X,y = X.to(device), y.to(device)

# forward pass
modelOutput = model(X)

# check the sizes
print(f'Model input is of size: {X.shape}')
print(f'Target output is of size: {y.shape}')
print(f'Model output is of size: {modelOutput.shape}')

# loss function
loss = loss_function(modelOutput,y[:,-1])

# extract a copy of the pretrained embedding weights for comparison later
pretrained_embeddings = model.embeddings.weight.detach().cpu().clone()

model,total_loss = train_the_model(model, device, dataloader, optimizer, loss_function, num_epochs=10)

postrained_embeddings = model.embeddings.weight.detach().cpu().clone()