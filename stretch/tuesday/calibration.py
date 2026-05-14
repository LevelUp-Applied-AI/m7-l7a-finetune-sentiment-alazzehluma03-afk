"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from manual_eval import manual_predict, compute_classification_report_from_arrays

def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    # bin edges via np.linspace(0, 1, n_bins + 1)
    # bucket_centers = midpoints of edges
    # for each prediction, take the max probability and the predicted class index
    #  assign each prediction to a bucket by its max probability
    #  bucket_accuracy = mean of (predicted == true) within the bucket; nan or 0 if empty
    #  bucket_count = number of predictions in the bucket
    #  return three numpy arrays
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        num_in_bin = np.sum(in_bin)
        bucket_counts[i] = num_in_bin
        
        if num_in_bin > 0:
            bucket_accuracies[i] = np.mean(predictions[in_bin] == y_true[in_bin])
        else:
            bucket_accuracies[i] = 0.0
            
    return bin_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    #  bucket predictions as in reliability_diagram
    # for each bucket, compute confidence (mean max probability) and accuracy
    #  weight |accuracy - confidence| by bucket fraction; sum
    # return float
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    n_total = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        bucket_size = np.sum(in_bin)
        
        if bucket_size > 0:
            bucket_accuracy = np.mean(predictions[in_bin] == y_true[in_bin])
            bucket_confidence = np.mean(confidences[in_bin])
            ece += (bucket_size / n_total) * np.abs(bucket_accuracy - bucket_confidence)
            
    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
def main():
    # 1. Setup Data
    # In a real scenario, these come from manual_predict() in your first file
    # Here are dummy arrays to test your logic:
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    
    # Simulated probabilities for 2 classes (shape: N, 2)
    probs = np.array([
        [0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3],
        [0.1, 0.9], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7], [0.8, 0.2]
    ])

    # 2. Run Calibration Calculations
    print("Calculating Reliability Diagram bins...")
    centers, accs, counts = reliability_diagram(probs, y_true, n_bins=5)
    
    print("Calculating Expected Calibration Error (ECE)...")
    ece_score = expected_calibration_error(probs, y_true, n_bins=5)
    
    # 3. Output Results
    print("\n" + "="*30)
    print("CALIBRATION RESULTS")
    print("="*30)
    print(f"ECE Score: {ece_score:.4f}")
    print("-" * 30)
    for i in range(len(centers)):
        print(f"Bin {i+1} (Center {centers[i]:.2f}): Count={int(counts[i])}, Accuracy={accs[i]:.4f}")
    print("="*30)

    # 4. Generate Plot
    output_dir = "stretch/tuesday/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "reliability_diagram.png")
    print(f"Saving plot to {output_path}...")
    plot_reliability(centers, accs, counts, output_path)

if __name__ == "__main__":
    main()