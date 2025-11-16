import torch
from torchinfo import summary

from src.llms.transformer.multi_head_attention.multi_head_attention_model import MultiHeadAttentionLanguageModel
# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device = ", device)

# ///////////////////////////////////////////////////////////////////////////////
# Hyperparameters
# ///////////////////////////////////////////////////////////////////////////////

# hyperparameters for GPT2-124M
vocab_size = 50257  # GPT2 vocab size
embedding_dimension = 768  # embedding dimension
sequence_length = 1024  # max sequence length
number_of_attention_heads = 12  # attention heads
num_blocks = 12  # transformer blocks
batch_size = 8

# ///////////////////////////////////////////////////////////////////////////////
# Create an instance and test it out
# ///////////////////////////////////////////////////////////////////////////////


model = MultiHeadAttentionLanguageModel(vocab_size, embedding_dimension, sequence_length, num_blocks,
                                        number_of_attention_heads, device).to(device)
print(model)
# run some fake data through
data = torch.randint(0, vocab_size, size=(batch_size, sequence_length)).to(device)
out = model(data)
print(f'Input size:  {data.shape}')
print(f'Output size: {out.shape}')

# ///////////////////////////////////////////////////////////////////////////////
# How many parameters do we have?
# ///////////////////////////////////////////////////////////////////////////////

# summary of model and parameters
summary(model, input_data=data, col_names=['input_size', 'output_size', 'num_params'])
