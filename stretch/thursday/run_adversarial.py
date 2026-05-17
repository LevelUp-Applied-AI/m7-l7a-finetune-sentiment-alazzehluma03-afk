"""
Stretch Thursday — Adversarial Evaluation.
Load a fine-tuned classifier, run it against adversarial_set.csv, and write results.csv.
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """Load model and tokenizer from local path."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """Run the model on every row and return full results DataFrame."""
    df = pd.read_csv(adv_csv_path)
    id2label = model.config.id2label
    
    results = []
    model.eval()
    
    for _, row in df.iterrows():
        inputs = tokenizer(row['text'], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)
            
        pred_label = id2label[pred_idx.item()]
        results.append({
            "predicted_label": pred_label,
            "predicted_probability": round(conf.item(), 4),
            "correct": pred_label.lower() == row['expected_label'].lower()
        })
    
    return pd.concat([df, pd.DataFrame(results)], axis=1)


def main() -> None:
    """Orchestrate and save results."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = "stretch/thursday/adversarial_set.csv"
    
    model, tokenizer = load_model(model_path)
    results_df = run_against_set(adv_csv, model, tokenizer)
    results_df.to_csv("stretch/thursday/results.csv", index=False)
    print("Evaluation complete. Results saved to stretch/thursday/results.csv")


if __name__ == "__main__":
    main()