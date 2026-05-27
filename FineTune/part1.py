import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# pip install torch transformers datasets peft accelerate bitsandbytes sentencepiece

# ============================================
# CONFIG
# ============================================

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "my_data.jsonl"
OUTPUT_DIR = "./finetuned-tinyllama-1.1b-chat"

# ============================================
# LOAD TOKENIZER
# ============================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Llama models usually don't have pad token
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# LOAD MODEL
# ============================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    #load_in_8bit=torch.cuda.is_available(),  # use 8bit only on GPU
)

model.config.pad_token_id = tokenizer.pad_token_id

# ============================================
# LoRA CONFIG
# ============================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Show trainable params
model.print_trainable_parameters()

# ============================================
# LOAD DATASET
# ============================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

# ============================================
# PREPROCESS
# ============================================

def format_example(example):

    prompt = (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)

# ============================================
# TRAINING ARGS
# ============================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-4,

    logging_steps=10,
    save_steps=200,

    fp16=torch.cuda.is_available(),

    report_to="none",

    save_total_limit=2,
)

# ============================================
# TRAINER
# ============================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)

# ============================================
# TRAIN
# ============================================

trainer.train()

# ============================================
# SAVE MODEL
# ============================================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining completed.")
print(f"Model saved to: {OUTPUT_DIR}")