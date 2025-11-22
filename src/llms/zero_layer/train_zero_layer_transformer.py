import torch
from transformers import AutoTokenizer

from src.llms.zero_layer.zero_layer_transformer import ZeroLayerTransformer


def pretty_print_tokens(token_ids: torch.Tensor, tokenizer: AutoTokenizer):
    """
    Pretty print token IDs and their corresponding strings.

    Args:
        token_ids (torch.Tensor): Tensor of token IDs.
        tokenizer (AutoTokenizer): Tokenizer to decode token IDs.
    """
    for i, token_id in enumerate(token_ids[0]):
        token_str = tokenizer.decode([token_id.item()])
        print(f"Token {i}: ID={token_id.item()}, String='{token_str}'")


# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = ZeroLayerTransformer(vocab_size=tokenizer.vocab_size, model_dimension=128)

# Tokenize some training data
text = "Mistakes are the portals of discovery"
token_ids = tokenizer.encode(text, return_tensors="pt")
print("Token IDs:")
pretty_print_tokens(token_ids, tokenizer)

# Compute the logits
logits = model(token_ids)
print(f"\nLogits Shape: [ {"×".join(str(x) for x in logits.shape)} ] (Batch Size x Sequence Length x Vocab Size)")
print(f"Logits:\n{logits.detach().numpy()}")

# Shift the logits to the right by one position
shifted_logits = logits[:, :-1, :]
print(f"\nShifted Logits Shape: [ {"×".join(str(x) for x in shifted_logits.shape)} ]")
print(f"Shifted Logits:\n{shifted_logits.detach().numpy()}")

# Shift the Token IDs to the left by one position
shifted_token_ids = token_ids[..., 1:]
print("\nShifted Token IDs:")
pretty_print_tokens(shifted_token_ids, tokenizer)

loss = torch.nn.functional.cross_entropy(
    shifted_logits.reshape(-1, logits.size(-1)),
    shifted_token_ids.reshape(-1)
)
print(f"\nLoss: {loss.item():.2f}")
