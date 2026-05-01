# ============================================================
# HAUSA CRISIS SIGNAL DETECTOR — TRAINING SCRIPT
# Run this in Google Colab (free GPU)
# ============================================================
# STEP 1: In Colab, go to Runtime > Change Runtime Type > GPU
# STEP 2: Upload hausa_crisis_data.csv to Colab files
# STEP 3: Run each cell in order
# ============================================================

# --- CELL 1: Install dependencies ---
# !pip install transformers datasets scikit-learn pandas torch accelerate -q

# --- CELL 2: Imports ---
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- CELL 3: Load and prepare data ---
df = pd.read_csv("hausa_crisis_data.csv")

# Define label mappings
LABELS = ["conflict", "displacement", "disease_outbreak", "flood", "food_insecurity", "no_crisis"]
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for idx, label in enumerate(LABELS)}

df["label_id"] = df["label"].map(label2id)
print("Label distribution:")
print(df["label"].value_counts())
print(f"\nTotal samples: {len(df)}")

# --- CELL 4: Split data ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

# --- CELL 5: Load AfriBERTa tokenizer and model ---
MODEL_NAME = "castorini/afriberta_large"  # Best for Hausa
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)
print("Model loaded successfully!")

# --- CELL 6: Tokenize datasets ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=128
    )

train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- CELL 7: Define metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# --- CELL 8: Training arguments ---
training_args = TrainingArguments(
    output_dir="./hausa_crisis_model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# --- CELL 9: Train ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training complete!")

# --- CELL 10: Evaluate ---
results = trainer.evaluate()
print("\nEvaluation Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")

# Detailed classification report
predictions = trainer.predict(test_tokenized)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=LABELS))

# --- CELL 11: Save model locally ---
model.save_pretrained("./hausa_crisis_model_final")
tokenizer.save_pretrained("./hausa_crisis_model_final")
print("Model saved to ./hausa_crisis_model_final")

# --- CELL 12: Push to Hugging Face Hub ---
# First run: !pip install huggingface_hub -q
# Then login: from huggingface_hub import notebook_login; notebook_login()
# Then uncomment below and replace YOUR_USERNAME:

model.push_to_hub("Skilgori/hausa-crisis-signal-detector")
tokenizer.push_to_hub("Skilgori/hausa-crisis-signal-detector")
print("Model pushed to Hugging Face Hub!")

# ============================================================
# AFTER TRAINING:
# 1. Download the hausa_crisis_model_final folder from Colab
# 2. Or push to HuggingFace Hub using the cell above
# 3. Then deploy the app.py file to Hugging Face Spaces
# ============================================================
