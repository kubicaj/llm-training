import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "distilgpt2"
LORA_MODEL = "./lora-quote-model"

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# BASE MODEL (NOT TRAINED)
# -----------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
).to(device)

prompt = "Quote by Albert Einstein:\n"

print("\n===== BEFORE TRAINING =====")
print(generate(base_model, tokenizer, prompt))

# -----------------------------
# LORA FINETUNED MODEL
# -----------------------------
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
).to(device)

lora_model = PeftModel.from_pretrained(lora_model, LORA_MODEL)

print("\n===== AFTER TRAINING =====")
print(generate(lora_model, tokenizer, prompt))
