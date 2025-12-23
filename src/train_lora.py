import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------------------------------------
# STEP 1 — Load & prepare dataset
# -----------------------------------------------------------
def load_shipcube_dataset(path):
    df = pd.read_csv(path)

    # Detect columns flexibly
    query_col = next((c for c in df.columns if "query" in c.lower()), None)
    response_col = next((c for c in df.columns if "response" in c.lower()), None)
    tone_col = next((c for c in df.columns if "tone" in c.lower()), None)
    thread_col = next((c for c in df.columns if "thread" in c.lower()), None)

    if not query_col or not response_col:
        raise ValueError("Dataset must contain query and response columns!")

    samples = []
    for _, row in df.iterrows():
        query = str(row[query_col])
        response = str(row[response_col])

        tone = str(row[tone_col]) if tone_col else "neutral"
        thread = str(row[thread_col]) if thread_col else ""

        # Final clean training text (NO JSON)
        text = (
            f"User Query: {query}\n"
            f"Tone: {tone}\n"
            f"Conversation Thread: {thread}\n\n"
            f"Assistant Response: {response}"
        )

        samples.append({"text": text})

    return Dataset.from_list(samples)


# -----------------------------------------------------------
# STEP 2 — Load model with 4-bit quantization
# -----------------------------------------------------------
def load_phi3_model():
    model_name = "microsoft/phi-2"  

    print("📌 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("📌 Loading 4-bit model…")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config
    )

    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer


# -----------------------------------------------------------
# STEP 3 — Train
# -----------------------------------------------------------
def train_model(data_path, output_path, epochs=2, batch_size=2, lr=2e-4):

    ds = load_shipcube_dataset(data_path)

    model, tokenizer = load_phi3_model()

    print("📌 Tokenizing…")
    def tokenize(example):
        enc = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(tokenize)

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to="none"
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator
    )

    print("🚀 Training starts…")
    trainer.train()

    print("💾 Saving model…")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✅ Training complete!")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()

    train_model(args.data, args.out, args.epochs, args.batch_size, args.lr)
