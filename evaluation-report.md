# Module 7 Week A — Lab Evaluation Report

## Dataset

The AARSynth App Reviews dataset contains 7,472 user reviews collected from 9 mobile applications, labelled with three sentiment classes: negative (0), neutral (1), and positive (2). The dataset is split 80/20 into 5,977 training examples and 1,495 test examples using a fixed seed of 42. Class distribution is approximately balanced: 2,519 negative, 2,442 neutral, and 2,511 positive examples.

## Model and hyperparameters

- **Backbone:** distilbert-base-uncased
- **Number of labels:** 3 (negative, neutral, positive)
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **Max length:** 128 tokens
- **Seed:** 42
- **Training time:** ~38 minutes (CPU, Windows, no GPU)

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---|
| Accuracy | 0.6375 |
| Macro-F1 | 0.6380 |

Per class:

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Negative | 0.7130 | 0.7382 | 0.6894 |
| Neutral  | 0.5070 | 0.4712 | 0.5486 |
| Positive | 0.6940 | 0.7245 | 0.6660 |

The neutral class is noticeably weaker than the other two, with an F1 of 0.507 compared to 0.713 and 0.694 for negative and positive. This is consistent with neutral being an inherently ambiguous label — reviews that are mildly positive or mildly negative can easily fall on either side of the boundary.

## Confusion matrix

|          | negative | neutral | positive |
|----------|----------|---------|----------|
| **negative** | 344 | 136 | 19 |
| **neutral**  | 93  | 254 | 116 |
| **positive** | 29  | 149 | 355 |

The matrix confirms that most errors involve the neutral class: 136 negative examples were predicted as neutral and 93 neutral examples were predicted as negative, suggesting the model struggles most with the negative–neutral boundary. Similarly, 149 positive examples were misclassified as neutral.

## Three qualitative error examples

### Example 1 — Gold: negative | Predicted: positive (confidence 0.8743)

> *"I have had this app for a while and it works really well."*

- **Gold label:** negative
- **Predicted label:** positive
- **Gold-class probability:** 0.0147

The surface language here is unambiguously positive — "works really well" is a strong positive signal. The negative gold label likely reflects context that is absent from the text itself (for example, a low star rating or a follow-up complaint in the original review that was not included in this excerpt). The model is not wrong given what it can see; the annotation may rely on out-of-text context.

### Example 2 — Gold: neutral | Predicted: negative (confidence 0.9732)

> *"This app doesn't want to sign me up."*

- **Gold label:** neutral
- **Predicted label:** negative
- **Gold-class probability:** 0.0204

The phrase "doesn't want to sign me up" describes a functional failure, which the model confidently treats as negative. A human annotator likely labelled it neutral because the tone is matter-of-fact rather than frustrated. The word "doesn't" combined with an unmet expectation strongly activates the model's negative class, making this a hard case for any surface-form classifier.

### Example 3 — Gold: positive | Predicted: negative (confidence 0.9651)

> *"It is a waste app and waste of time. It is only useful for downloading the movies."*

- **Gold label:** positive
- **Predicted label:** negative
- **Gold-class probability:** 0.0082

This is a genuinely ambiguous sentence. The opening is strongly negative ("waste app", "waste of time") and the model focuses on those cues. The positive interpretation hinges on reading "only useful for downloading movies" as high praise for the app's core purpose — a nuanced reading that requires understanding the user's intent. The model overweights the explicit negative phrases in the first clause, which is a reasonable failure mode for a 128-token window model without discourse understanding.

## Hugging Face Hub model URL

https://huggingface.co/Alazzehluma-21/m7-app-review-sentiment
