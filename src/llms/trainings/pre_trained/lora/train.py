import torch
import logging
import math

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

# =======================================================
# 1️⃣ LOGGING SETUP
# =======================================================
# Python logging for high-level events.
# Hugging Face Trainer prints progress bars separately.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =======================================================
# 2️⃣ BASIC CONFIGURATION
# =======================================================
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./lora-quote-model"

MAX_LENGTH = 128          # Max tokens per training example
BATCH_SIZE = 8            # Batch size per GPU
EPOCHS = 10               # Upper bound (early stopping may stop earlier)
LR = 2e-4                 # Learning rate (LoRA parameters only)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# =======================================================
# 3️⃣ LOAD & SPLIT DATASET
# =======================================================
logger.info("Loading dataset: Abirate/english_quotes")

# Dataset only has "train", so we split manually
raw_dataset = load_dataset("Abirate/english_quotes")["train"]

# 90% training / 10% validation
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

logger.info(f"Training examples: {len(dataset['train'])}")
logger.info(f"Validation examples: {len(dataset['test'])}")

def format_example(example):
    """
    Convert dataset fields into a single text prompt.

    IMPORTANT:
    The model sees ONLY this text.
    There is no notion of "author" or "quote" fields anymore.
    """
    return {
        "text": f"Quote by {example['author']}:\n{example['quote']}"
    }

# Apply formatting and remove original dataset columns
dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

# =======================================================
# 4️⃣ TOKENIZATION
# =======================================================
logger.info("Loading tokenizer")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-style models do not define a PAD token
# We reuse EOS so padding is safe
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    """
    Convert text into:
    - input_ids
    - attention_mask

    Padding ensures equal-length tensors for batching.
    """
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

# Tokenize both training and validation splits
tokenized_ds = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],  # CRITICAL: remove raw strings
)

# =======================================================
# 5️⃣ LOAD BASE MODEL (BF16)
# =======================================================
logger.info("Loading base GPT model")

# BF16:
# - Same memory usage as FP16
# - Much more numerically stable
# - No GradScaler required
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

# =======================================================
# 6️⃣ APPLY LoRA ADAPTERS
# =======================================================
logger.info("Applying LoRA adapters")

lora_config = LoraConfig(
    r=8,
    # Rank of LoRA matrices (capacity vs memory)

    lora_alpha=16,
    # Scaling factor applied to LoRA updates

    target_modules=["c_attn", "c_proj"],
    # GPT-2 attention projection layers

    lora_dropout=0.05,
    # Regularization (important for small datasets)

    bias="none",
    # Bias parameters remain frozen

    task_type="CAUSAL_LM",
    # Autoregressive language modeling

    fan_in_fan_out=True,
    # REQUIRED for GPT-style Conv1D layers
)

# Attach LoRA adapters to the frozen base model
model = get_peft_model(model, lora_config)

# Print how many parameters are actually trained
model.print_trainable_parameters()

# =======================================================
# 7️⃣ DATA COLLATOR
# =======================================================
# Automatically creates labels by shifting input_ids.
# This is standard causal language modeling.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =======================================================
# 8️⃣ PERPLEXITY CALLBACK (CORRECT WAY)
# =======================================================
class PerplexityCallback(TrainerCallback):
    """
    Hugging Face does NOT pass loss to compute_metrics()
    for causal language models.

    Instead, eval_loss is already computed internally.
    This callback converts eval_loss → perplexity safely.
    """

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])

# =======================================================
# 9️⃣ TRAINING ARGUMENTS
# =======================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # -------- Training --------
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    # Effective batch size = 16

    learning_rate=LR,
    num_train_epochs=EPOCHS,

    # -------- Precision --------
    bf16=True,
    fp16=False,

    # -------- Evaluation & Saving --------
    eval_strategy="epoch",
    save_strategy="epoch",

    # Save ONLY the best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,

    # -------- Logging --------
    logging_steps=50,
    report_to="none",

    # -------- Optimization --------
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # -------- Misc --------
    remove_unused_columns=False,
)

# =======================================================
# 🔟 EARLY STOPPING
# =======================================================
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,
    # Stop training if validation loss does not improve
    # for 2 consecutive evaluation rounds (epochs)
)

# =======================================================
# 1️⃣1️⃣ TRAINER
# =======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=data_collator,
    callbacks=[
        early_stopping,
        PerplexityCallback(),
    ],
)

# =======================================================
# 1️⃣2️⃣ START TRAINING
# =======================================================
logger.info("Starting training")
trainer.train()

# =======================================================
# 1️⃣3️⃣ SAVE FINAL (BEST) MODEL
# =======================================================
# Because load_best_model_at_end=True,
# this saves the BEST checkpoint, not the last epoch.
logger.info("Saving best LoRA fine-tuned model")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Training completed successfully")
