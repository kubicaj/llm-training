import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "distilgpt2"
LORA_PATH = "./qlora-quote-model"

device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.8,
            do_sample=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# -------------------------
# BEFORE (base model)
# -------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("\n===== BEFORE QLoRA TRAINING =====")
print(generate(base_model, "Quote by Albert Einstein:\n"))

# -------------------------
# AFTER (QLoRA)
# -------------------------
qlora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("\n===== AFTER QLoRA TRAINING =====")
print(generate(qlora_model, "Quote by Albert Einstein:\n"))
