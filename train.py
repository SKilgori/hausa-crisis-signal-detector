# ============================================================
# HAUSA CRISIS SIGNAL DETECTOR — TRAINING SCRIPT
# Run this in Google Colab (free GPU)
# ============================================================
# STEP 1: In Colab, go to Runtime > Change Runtime Type > GPU
# STEP 2: Clone or upload the repo so ./data/hausa_crisis_data.csv exists
#         relative to wherever you run this from (see DATA_PATH below —
#         previously this pointed at a bare "hausa_crisis_data.csv" which
#         silently resolved to a stale 39-row stub file at the repo root
#         instead of the real 200-row dataset in data/. Fixed here.)
# STEP 3: Run each cell in order
# ============================================================

# --- CELL 1: Install dependencies ---
# !pip install transformers datasets scikit-learn pandas torch accelerate -q

# --- CELL 2: Imports ---
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- CELL 3: Load and prepare data ---
# FIX: explicit data/ path — was a bare filename that resolved to a stale
# 39-row stub at the repo root instead of the real dataset.
DATA_PATH = "data/hausa_crisis_data.csv"
df = pd.read_csv(DATA_PATH)

# FIX: dedup as a defensive safety net. The committed CSV should already be
# deduped (see accompanying data file), but this guards against future data
# additions accidentally reintroducing exact-duplicate rows that could leak
# across the train/test split.
before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
if before != len(df):
    print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} remaining")

LABELS = ["conflict", "displacement", "disease_outbreak", "flood", "food_insecurity", "no_crisis"]
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for idx, label in enumerate(LABELS)}

df["label_id"] = df["label"].map(label2id)
print("Label distribution:")
print(df["label"].value_counts())
print(f"\nTotal samples: {len(df)}")

if len(df) < 150:
    print(
        f"\nWARNING: only {len(df)} examples across {len(LABELS)} classes "
        f"(~{len(df)//len(LABELS)}/class). Treat resulting accuracy/F1 as a "
        f"first-pass baseline, not a production guarantee."
    )

# --- CELL 4: Split data ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False
)
test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False
)

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
    processing_class=tokenizer,  # current HF-recommended param, replaces deprecated tokenizer=
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

# Detailed per-class report — check this, not just the headline numbers above.
predictions = trainer.predict(test_tokenized)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=LABELS, digits=3))

# --- CELL 11: Save model locally ---
OUTPUT_DIR = "./hausa_crisis_model_final"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(f"{OUTPUT_DIR}/label_mapping.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
print(f"Model saved to {OUTPUT_DIR}")

# --- CELL 12: Push to Hugging Face Hub (OPTIONAL, OFF by default) ---
# FIX: previously this ran unconditionally with no login step actually
# executed, which would crash. It also published the model PUBLICLY by
# default — anyone could then download and self-host it for free, which
# works against using this as a paid API (see api.py). Set PUSH_TO_HUB=True
# only if you deliberately want this, and consider a private/gated repo
# instead of a fully public one if the FastAPI/RapidAPI path is the goal.
PUSH_TO_HUB = False

if PUSH_TO_HUB:
    from huggingface_hub import notebook_login
    print("Please log in to Hugging Face to push the model.")
    notebook_login()

    REPO_ID = "SKilgori/hausa-crisis-signal-detector"  # matches GitHub casing
    model.push_to_hub(REPO_ID)
    tokenizer.push_to_hub(REPO_ID)
    print("Model pushed to Hugging Face Hub!")
else:
    print(f"PUSH_TO_HUB is False — skipping. Model is available locally at {OUTPUT_DIR}")

# ============================================================
# AFTER TRAINING:
# 1. Download the hausa_crisis_model_final folder from Colab (or copy into
#    Drive — Colab's local disk is ephemeral and won't survive a runtime
#    disconnect)
# 2. For the Gradio/HF Spaces demo: push to the Hub (set PUSH_TO_HUB=True)
#    and update MODEL_PATH in app.py
# 3. For the FastAPI/RapidAPI service: keep it local, point api.py's
#    MODEL_PATH at this folder instead
# ============================================================
