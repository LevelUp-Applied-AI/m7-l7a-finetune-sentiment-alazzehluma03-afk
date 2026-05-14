# Calibration Analysis

— fill each section after running your manual evaluation and reliability diagram.

## Reliability diagram interpretation

What does your saved diagram (`figures/reliability-diagram.png`) look like? Where is the model over-confident vs. under-confident? Cite specific bucket values.

The saved diagram (figures/reliability-diagram.png) shows three distinct bars corresponding to confidence levels of 0.5, 0.7, and 0.9. In all three cases, the empirical accuracy is 1.0 (100%). The model is strictly under-confident; for example, in the 0.5 bin, the model is correct 100% of the time despite only claiming 50% confidence. There are no instances of over-confidence (where bars would fall below the diagonal dashed line).

## Expected Calibration Error

Report your ECE. Interpret what it says about model trustworthiness for production use.

The reported ECE is 0.2300.
This means that, on average, there is a 23% gap between what the model predicts as its probability and its actual accuracy. For production use, this suggests that the model’s "uncertainty" is not a reliable metric. While the model is highly accurate, it is being far too conservative with its probability scores, making it difficult to use these scores for risk-sensitive decision-making.

## A specific calibration pattern

Identify one specific pattern (over-confidence on majority class, under-confidence near boundaries, etc.) and reason about why it arose given how the model was trained.

The diagram identifies a systematic under-confidence across all confidence buckets. This likely arose because the model was fine-tuned on a task that it found relatively easy (leading to high accuracy), but the training process (or hyperparameters like weight decay/learning rate) was not aggressive enough to push the Softmax outputs to the extremes. The model successfully learned to separate the classes, but the internal "probability" hasn't yet caught up to its external success.

## A proposed engineering action

What would you change in production based on these findings? (Threshold-based abstention, temperature scaling, bucket-specific data collection, etc.)

Based on these findings, I propose implementing Temperature Scaling in production. Since the model's accuracy is already perfect (1.0) but its confidence is lagging, we can divide the logits by a temperature parameter
T < 1 before the final Softmax. This will "stretch" the probabilities toward 1.0, effectively shifting the bars in the reliability diagram to the right to better align with the diagonal line of perfect calibration.
