import torch


from transformers import GPT2Tokenizer

from src.llms.transformer.one_head_attention.one_head_attention_model import OneHeadAttentionLanguageModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# ///////////////////////////////////////////////////////////////////////////////
# Hyperparameters
# ///////////////////////////////////////////////////////////////////////////////

# data hyperparameters
seq_len = 8 # aka context window
n_vocab = tokenizer.vocab_size

# model hyperparameters
embed_dim = 128
nTransformerBlocks = 12

# training hyperparameters
batch_size = 5

# ///////////////////////////////////////////////////////////////////////////////
# Create a model instance and inspect
# ///////////////////////////////////////////////////////////////////////////////

llm = OneHeadAttentionLanguageModel(nTransformerBlocks, n_vocab, embed_dim, seq_len)
print(llm)

# create data
tokens = tokenizer.encode('I prefer oat milk in my coffee.')
X = torch.tensor(tokens[:-1]).unsqueeze(0)
y = torch.tensor(tokens[1:]).unsqueeze(0)

print(X.shape)
print(y.shape)

out = llm(X)

print(out.shape)

# Generate new tokens (for example, 10 new tokens)
generated_tokens = llm.generate(X, num_new_tokens=10)

# Remove batch dimension and convert to list
generated_token_ids = generated_tokens[0].tolist()

# Decode to text
generated_text = tokenizer.decode(generated_token_ids)

print(generated_text)


