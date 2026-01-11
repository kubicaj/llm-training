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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# =======================================================
# 1️⃣ LOGGING SETUP
# =======================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =======================================================
# 2️⃣ BASIC CONFIGURATION
# =======================================================
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./qlora-quote-model"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 10              # early stopping will stop sooner
LR = 2e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# =======================================================
# 3️⃣ LOAD & SPLIT DATASET
# =======================================================
logger.info("Loading dataset: Abirate/english_quotes")

raw_dataset = load_dataset("Abirate/english_quotes")["train"]
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

logger.info(f"Train: {len(dataset['train'])}")
logger.info(f"Val:   {len(dataset['test'])}")

def format_example(example):
    return {
        "text": f"Quote by {example['author']}:\n{example['quote']}"
    }

dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

# =======================================================
# 4️⃣ TOKENIZER
# =======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

tokenized_ds = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

# =======================================================
# 5️⃣ LOAD BASE MODEL (QLoRA: 4-bit NF4)
# =======================================================
logger.info("Loading base model in 4-bit (QLoRA)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# =======================================================
# 6️⃣ APPLY LoRA (same as before)
# =======================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =======================================================
# 7️⃣ DATA COLLATOR
# =======================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =======================================================
# 8️⃣ PERPLEXITY CALLBACK
# =======================================================
class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])

# =======================================================
# 9️⃣ TRAINING ARGUMENTS (QLoRA-specific)
# =======================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    num_train_epochs=EPOCHS,

    bf16=True,
    fp16=False,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,

    logging_steps=50,
    report_to="none",

    # 🔑 IMPORTANT FOR QLoRA
    optim="paged_adamw_8bit",

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    remove_unused_columns=False,
)

# =======================================================
# 🔟 EARLY STOPPING
# =======================================================
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2
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
    callbacks=[early_stopping, PerplexityCallback()],
)

# =======================================================
# 1️⃣2️⃣ TRAIN
# =======================================================
logger.info("Starting QLoRA training")
trainer.train()

# =======================================================
# 1️⃣3️⃣ SAVE BEST MODEL
# =======================================================
logger.info("Saving best QLoRA adapter")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Training completed successfully")
