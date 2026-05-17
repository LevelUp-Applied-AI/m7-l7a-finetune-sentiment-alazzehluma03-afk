# Adversarial Evaluation Analysis

— fill each section after running `run_adversarial.py` against your fine-tuned model.

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 2 | 6 | 33% |
| lexical_trigger | 3 | 5 | 60% |
| domain_shift | 4 | 4 | 100% |
| length_extreme | 4 | 5 | 80% |
| sarcasm | 1 | 5 | 20% |
| other | 3 | 3 | 100% |

## Confirmed hypotheses

Which categories did the model fail on as you predicted? Cite specific row IDs and predictions.

The model failed as predicted on **sarcasm** and **negation**. 

- In sarcasm (rows 14, 15, 23, 27), it usually ignored the ironic intent and picked up the positive surface words, predicting positive instead of negative.
- In negation (rows 1, 18, 25), it struggled with phrases like "did not crash", "not slow at all", and "I can't complain", often defaulting to neutral.


## Refuted hypotheses

Which categories did the model handle better than you expected?

The model performed better than expected on **domain_shift** and **length_extreme**. It handled all 4 domain shift examples correctly (phone reviews, restaurant, movie, Windows update) and did well on both very short and very long reviews.

## What the results reveal about the decision boundary

Articulate one or more specific things the adversarial results say about how the model decides.

The model relies heavily on individual positive or negative keywords rather than overall sentence meaning or context. This explains why it breaks on sarcasm and subtle negation but works fine when the polarity words are clear and direct.