"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    #  iterate texts in batches
    # tokenize each batch with truncation, max_length=128, padding=True, return_tensors='pt'
    # forward pass under torch.no_grad()
    # softmax over the last dim
    #  argmax to get class indices
    # collect into numpy arrays of shape (N,) and (N, num_classes); return both
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_preds = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, truncation=True, max_length=128, padding=True, return_tensors='pt').to(device)
        with torch.no_grad(): # Disable gradient calculation for speed/memory [cite: 8]
            outputs = model(**inputs)
            logits = outputs.logits # Raw unnormalized scores [cite: 9, 24]
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Move to CPU and store [cite: 22, 54]
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)
def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    #  compute true positives / false positives / false negatives per class
    #  precision = TP / (TP + FP); guard divide-by-zero
    #  recall = TP / (TP + FN)
    #  f1 = 2 * P * R / (P + R)
    #  accuracy = sum(y_pred == y_true) / N
    #  macro-F1 = mean of per-class f1 scores
    #  assemble and return the dict
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    labels = np.unique(y_true)
    per_class = {}
    f1_scores = []
    for label in labels:
      tp = np.sum((y_pred == label) & (y_true == label))
      fp = np.sum((y_pred == label) & (y_true != label))
      fn = np.sum((y_pred != label) & (y_true == label))
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
      else:
            f1 = 0.0
            
      per_class[int(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
      f1_scores.append(f1)
    macro_f1 = np.mean(f1_scores) if len(f1_scores) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }
def main():
  model_path = "distilbert-base-uncased"
  test_texts = [
        "The movie was great and I enjoyed it.",
        "It was a terrible experience, very disappointed.",
        "I don't know how to feel about this, it was okay."
    ]
  y_true = [1, 0, 1]
  try:
    print(f"Loading tokenizer and model: {model_path}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Running manual inference loop...")
    preds, probs = manual_predict(model, tokenizer, test_texts)
    print("Computing metrics from primitives...")
    metrics = compute_classification_report_from_arrays(y_true, preds)
    print("\n" + "="*30)
    print("RESULTS FROM PRIMITIVES")
    print("="*30)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("-" * 30)
    for label, scores in metrics['per_class'].items():
      print(f"Class {label}: F1={scores['f1']:.4f}")
    print("="*30)
  except Exception as e:
      print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
      
