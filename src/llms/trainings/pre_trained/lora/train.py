import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# -------------------------------------------------------
# LOGGING SETUP (this controls training logs you see)
# -------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# BASIC CONFIGURATION
# -------------------------------------------------------
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./lora-quote-model"

MAX_LENGTH = 128          # Max tokens per training example
BATCH_SIZE = 8            # Batch size per GPU step
EPOCHS = 3                # Full passes over dataset
LR = 2e-4                 # Learning rate for LoRA weights only

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# -------------------------------------------------------
# LOAD DATASET (from Hugging Face automatically)
# -------------------------------------------------------
logger.info("Loading dataset: Abirate/english_quotes")
dataset = load_dataset("Abirate/english_quotes")

def format_example(example):
    """
    Converts raw dataset fields into a single text prompt.
    This is what the model actually learns from.
    """
    return {
        "text": f"Quote by {example['author']}:\n{example['quote']}"
    }

dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names
)

logger.info(f"Total training examples: {len(dataset['train'])}")

# -------------------------------------------------------
# TOKENIZER
# -------------------------------------------------------
logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT models have no pad token → reuse EOS
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    """
    Converts text into token IDs that the model understands.
    """
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

# -------------------------------------------------------
# LOAD BASE MODEL
# -------------------------------------------------------
logger.info("Loading base GPT model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,  # saves VRAM
    device_map="auto",
)

# -------------------------------------------------------
# LORA CONFIGURATION
# -------------------------------------------------------
logger.info("Applying LoRA adapters")

lora_config = LoraConfig(
    r=8,
    # Rank of LoRA matrices
    # Higher = more capacity, more VRAM

    lora_alpha=16,
    # Scaling factor for LoRA updates

    target_modules=["c_attn", "c_proj"],
    # Attention projection layers in GPT-2 architecture

    lora_dropout=0.05,
    # Prevents overfitting on small datasets

    bias="none",
    # Do not train bias parameters

    task_type="CAUSAL_LM",
    # Autoregressive language modeling,

    fan_in_fan_out=True,  # <-- ADD THIS
)

model = get_peft_model(model, lora_config)

# Print how many parameters are actually trainable
model.print_trainable_parameters()

# -------------------------------------------------------
# DATA COLLATOR
# -------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM, NOT masked LM
)

# -------------------------------------------------------
# TRAINING ARGUMENTS (IMPORTANT PART)
# -------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Where checkpoints & logs are stored

    per_device_train_batch_size=BATCH_SIZE,
    # Batch size PER GPU (RTX 12GB → safe value)

    gradient_accumulation_steps=2,
    # Simulates batch size = BATCH_SIZE * 2
    # Reduces GPU memory usage

    learning_rate=LR,
    # Learning rate ONLY for LoRA layers

    num_train_epochs=EPOCHS,
    # Number of times the full dataset is seen

    bf16=True,
    fp16=False,
    # Enables mixed precision → lower VRAM usage

    logging_steps=50,
    # Print loss every 50 steps

    save_strategy="epoch",
    # Save model after each epoch

    report_to="none",
    # Disable Weights & Biases / TensorBoard

    eval_strategy="no",
    # No validation set (simple demo)

    optim="adamw_torch",
    # Stable optimizer for transformer training

    lr_scheduler_type="cosine",
    # Smooth learning rate decay

    warmup_ratio=0.05,
    # First 5% steps slowly increase LR (stability)

    remove_unused_columns=False,
    # Required for PEFT models
)

# -------------------------------------------------------
# TRAINER
# -------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    data_collator=data_collator,
)

# -------------------------------------------------------
# START TRAINING
# -------------------------------------------------------
logger.info("Starting training")
trainer.train()

# -------------------------------------------------------
# SAVE FINAL MODEL
# -------------------------------------------------------
logger.info("Saving LoRA fine-tuned model")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Training completed successfully")
