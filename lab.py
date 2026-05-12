"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    The CSV must have at least `text` and `label` columns. (The curated
    `data/app_reviews_train.csv` also includes `app`, `app_name`, and `rating`
    columns — these are useful for inspection but not required by the model.)

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)
    raw_ds = Dataset.from_pandas(df, preserve_index=False)
    ds_split = raw_ds.train_test_split(test_size=test_size, seed=seed)
    return ds_split


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    `tokenizer` is a loaded HuggingFace tokenizer (callable) — load it once
    in `main()` via `AutoTokenizer.from_pretrained(...)` and pass it in.
    Use truncation=True and max_length=max_length. Do not pad here — padding is
    applied dynamically by DataCollatorWithPadding at training time.

    Note: this signature differs from the drill (`tokenize_dataset(ds, name)`)
    by accepting the loaded tokenizer object so `main()` doesn't re-load it.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
    tokenized_ds = ds_dict.map(tokenize_fn, batched=True)
    return tokenized_ds


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )
    args.eval_strategy = args.eval_strategy.value
    args.save_strategy = args.save_strategy.value
    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {
        "accuracy": acc,
        "macro_f1": f1,
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model). Pass
    id2label=ID2LABEL and label2id=LABEL2ID to the model so its config records
    the human-readable label names — Integration 7A reads them from
    `model.config.id2label` rather than hard-coding.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # FIX: use `tokenizer=` instead of `processing_class=`.
    # `processing_class` was added in transformers 4.46+; the course pins
    # transformers>=4.41,<5.0, so environments on 4.41–4.45 would crash.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Read label names from trainer.model.config.id2label (do not hard-code).

    Returns:
        {
            "accuracy":            float,
            "macro_f1":            float,
            "per_class_f1":        {label_name: float, ...},
            "per_class_precision": {label_name: float, ...},
            "per_class_recall":    {label_name: float, ...},
        }
    """
    from sklearn.metrics import precision_score, recall_score

    predictions_output = trainer.predict(tokenized_test)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    f1_none   = f1_score(labels, preds, average=None)
    prec_none = precision_score(labels, preds, average=None, zero_division=0)
    rec_none  = recall_score(labels, preds, average=None, zero_division=0)

    id2label = trainer.model.config.id2label
    per_class_f1        = {id2label[i]: float(f1_none[i])   for i in range(len(f1_none))}
    per_class_precision = {id2label[i]: float(prec_none[i]) for i in range(len(prec_none))}
    per_class_recall    = {id2label[i]: float(rec_none[i])  for i in range(len(rec_none))}

    return {
        "accuracy":            float(acc),
        "macro_f1":            float(f1_macro),
        "per_class_f1":        per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall":    per_class_recall,
    }


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path  = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    ds        = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV — all required columns + full softmax distribution
    pred_output = trainer.predict(tokenized["test"])
    pred_logits = pred_output.predictions
    pred_idx    = np.argmax(pred_logits, axis=1)
    pred_probs  = F.softmax(torch.tensor(pred_logits), dim=-1).numpy()
    id2label    = trainer.model.config.id2label

    results_dict = {
        "text":                 ds["test"]["text"],
        "label":                [id2label[i] for i in ds["test"]["label"]],
        "predicted_label":      [id2label[i] for i in pred_idx],
        "predicted_probability":[float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    }
    for i, label_name in id2label.items():
        results_dict[f"prob_{label_name}"] = pred_probs[:, i]

    df_out = pd.DataFrame(results_dict)
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix — print + persist as CSV
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    df_cm = pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values()))
    print(df_cm.to_string())
    df_cm.to_csv("confusion_matrix.csv")

    # Push to Hugging Face Hub (Task 5) — skipped in CI
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/Alazzehluma-21/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and try again.")


if __name__ == "__main__":
    main()
