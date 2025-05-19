import accelerate
from accelerate.data_loader import DataLoaderShard
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# Save original DataLoaderShard __init__
_orig_init = DataLoaderShard.__init__

def patched_init(self, dataset, use_stateful_dataloader=False, **kwargs):
    # Remove 'in_order' if present to avoid error
    if 'in_order' in kwargs:
        kwargs.pop('in_order')

    # Call original __init__ with cleaned kwargs
    _orig_init(self, dataset, use_stateful_dataloader=use_stateful_dataloader, **kwargs)

# Override __init__ with patched version
DataLoaderShard.__init__ = patched_init

print("Patched accelerate.data_loader.DataLoaderShard to remove 'in_order' argument.")

import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
from huggingface_hub import login
import os

# Login to Hugging Face Hub 
login(token="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") 

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit to save memory
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Print trainable params
model.print_trainable_parameters()

# Load and process your CSV
df = pd.read_csv("/home/3350501/bocconi-nlrl-main/finetune_train.csv")

def format_example(example):
    prompt = f"<s>[INST] <<SYS>>\nYou will answer and explain in a logical manner.\n<</SYS>>\n{example['instruction']} [/INST] {example['output'].strip()}{tokenizer.eos_token}"
    return {"text": prompt}

dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_example)

# Tokenization
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="/home/3350501/bocconi-nlrl-main/saved_models",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    fp16=True,
    bf16=False,
    report_to="none",
    save_total_limit=3
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save LoRA adapter only
model.save_pretrained("/home/3350501/bocconi-nlrl-main/saved_models")
tokenizer.save_pretrained("/home/3350501/bocconi-nlrl-main/saved_models")
