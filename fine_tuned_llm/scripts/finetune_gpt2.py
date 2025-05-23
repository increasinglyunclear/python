import json
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
from tqdm import tqdm

def load_and_prepare_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert the data into a format suitable for training
    texts = []
    for item in data:
        # Use the text field directly
        text = item['text']
        texts.append(text)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    return dataset

def main():
    # Initialize tokenizer and model
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset("data/processed_texts.json")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./models/gpt2_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=False,
        report_to=[],  # Disable wandb and other reporting
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained("./models/gpt2_finetuned")

if __name__ == "__main__":
    main() 