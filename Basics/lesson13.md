# 📊 ML Development Lifecycle — Part 8: Model Evaluation

We have now **trained** our model.

But training is only half the story.

The model says:

> “I learned!”

And we must respond:

> **“Okay… prove it.”** 😄

That is **model evaluation**.

```text id="4u7q2p"
Problem
   ↓
Data
   ↓
EDA
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
Model Selection
   ↓
Training
   ↓
⭐ Evaluation
   ↓
Deployment
```

---

# 1. What Is Model Evaluation?

Model evaluation means:

> **Measuring how well a trained model performs on data it has not seen during training.**

The important phrase is:

### **unseen data**

Because we don't care whether the model memorized its training examples.

We care whether it can **generalize**.

---

# 2. The Student Analogy 🎓

Imagine you teach a student 100 questions.

They memorize all 100.

Then you ask the exact same questions:

```text
Student → 100/100
```

Does that mean they're intelligent?

Not necessarily.

Now give them 20 new questions:

```text
Student → 17/20
```

Now we have evidence that they actually learned something.

ML is the same:

```text
Training Data
     ↓
Model learns
     ↓
Unseen Data
     ↓
Evaluate
```

---

# 3. Different Problems Need Different Metrics

This is extremely important.

```text id="1f8d0j"
              Model
                │
       ┌────────┴─────────┐
       ↓                  ↓
   Regression       Classification
       │                  │
       ↓                  ↓
 MAE/MSE/RMSE        Accuracy
 R²                  Precision
                     Recall
                     F1
                     ROC-AUC
```

There isn't one universal metric.

---

# PART A — REGRESSION METRICS

Suppose we're predicting:

```text
Taxi Fare
```

Actual:

```text
1000
```

Prediction:

```text
900
```

Error:

$$
1000-900=100
$$

---

# 4. MAE — Mean Absolute Error

MAE asks:

> **"On average, how far are my predictions from the real values?"**

Formula:

$$
MAE=\frac{1}{n}\sum |y_i-\hat y_i|
$$

Suppose:

```text
Actual       Prediction
100          90
200          220
300          280
```

Errors:

```text
10
20
20
```

Average:

$$
MAE=\frac{10+20+20}{3}
$$

$$
MAE=16.67
$$

So:

> The model is wrong by about **16.67 units on average**.

---

# 5. Why Absolute Value?

Suppose errors are:

```text
+100
-100
```

If we simply average:

$$
\frac{100-100}{2}=0
$$

It would falsely look perfect!

So MAE uses:

$$
|error|
$$

giving:

```text
100
100
```

---

# 6. MAE Mental Model

Think:

> **"How many units away am I?"**

For example:

```text
MAE = $500
```

means:

> Predictions are off by about $500 on average.

This makes MAE very intuitive.

---

# 7. MSE — Mean Squared Error

MSE squares the errors.

$$
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
$$

Why square?

Because large errors should hurt more.

Example:

```text
Error = 2
Squared = 4

Error = 10
Squared = 100
```

A large mistake becomes **much more expensive**.

---

# 8. MAE vs MSE

Suppose:

```text
Errors:

2
3
20
```

MAE:

$$
\frac{2+3+20}{3}=8.33
$$

MSE:

$$
\frac{4+9+400}{3}=137.67
$$

See what happened?

The error of 20 dominates MSE.

Therefore:

```text id="0vq5a6"
MAE
↓
Treats errors more evenly

MSE
↓
Punishes large errors heavily
```

---

# 9. RMSE

RMSE = Root Mean Squared Error.

$$
RMSE=\sqrt{MSE}
$$

Why take the square root?

Because MSE has squared units.

If target is:

```text
Dollars
```

MSE is:

```text
Dollars²
```

which isn't very intuitive.

RMSE returns to:

```text
Dollars
```

---

# 10. MAE vs RMSE

This is one of the most useful comparisons.

### MAE:

> "Average size of error."

### RMSE:

> "Average error, with **extra punishment for large mistakes**."

So:

```text
Small errors + occasional huge errors
        ↓
RMSE becomes large
```

---

# 11. When Should You Use MAE?

Use MAE when:

> You want an easy-to-understand measure of typical error and don't want huge errors to dominate as strongly.

Example:

```text
House price prediction
```

You might say:

> "Our predictions are off by $18,000 on average."

Very understandable.

---

# 12. When Should You Use RMSE?

Use RMSE when:

> Large errors are especially undesirable.

Example:

```text
Electricity demand prediction
```

A prediction that's slightly wrong is okay.

But a massive error could cause serious problems.

RMSE will emphasize those large mistakes.

---

# 13. R² — The Famous One

R² is called:

> **Coefficient of Determination**

It answers approximately:

> **"How much of the variation in the target does the model explain relative to a simple mean baseline?"**

A common formula is:

$$
R^2=1-\frac{SS_{res}}{SS_{tot}}
$$

---

# 14. R² Intuition

Suppose:

```text
R² = 0
```

Your model isn't improving over the baseline of predicting the mean, in the standard formulation.

If:

```text
R² = 0.80
```

we often say:

> The model explains about 80% of the variance relative to that baseline.

But be careful:

**R² is not simply "80% accuracy."**

---

# 15. Can R² Be Negative?

Yes!

This surprises beginners.

You can have:

$$
R^2 < 0
$$

That means the model performs worse than the simple mean-prediction baseline on the evaluated data.

So:

```text
R² = -0.5
```

🚨 Your model is doing badly.

---

# 16. Regression Metrics Summary

| Metric | Main Idea                        | Large Errors    |
| ------ | -------------------------------- | --------------- |
| MAE    | Average absolute error           | Less emphasis   |
| MSE    | Average squared error            | Strong emphasis |
| RMSE   | Error in original units          | Strong emphasis |
| R²     | Relative explanatory performance | Indirect        |

---

# PART B — CLASSIFICATION

Now suppose our model predicts:

```text
Spam
or
Not Spam
```

This is classification.

---

# 17. Accuracy

The simplest metric:

$$
Accuracy=\frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

Suppose:

```text
100 emails

95 correctly classified
```

Then:

$$
Accuracy=95\%
$$

Sounds excellent.

But…

# 🚨 Accuracy Can Lie

---

# 18. The Imbalanced Dataset Problem

Imagine detecting a rare disease.

Out of:

```text
10,000 people
```

Only:

```text
100 have disease
```

A terrible model predicts:

> "Nobody has the disease."

It gets:

```text
9,900 correct
100 wrong
```

Accuracy:

$$
99\%
$$

😱

99% accuracy sounds amazing.

But the model detected:

$$
0/100
$$

patients.

It's useless.

---

# 19. Confusion Matrix ⭐

To properly understand classification, we need the **confusion matrix**.

```text id="6d1rpx"
                    ACTUAL
                Positive   Negative
             ┌───────────┬───────────┐
Predicted    │           │           │
Positive     │    TP     │    FP     │
             │           │           │
             ├───────────┼───────────┤
Predicted    │           │           │
Negative     │    FN     │    TN     │
             │           │           │
             └───────────┴───────────┘
```

These four letters are extremely important.

---

# 20. True Positive — TP

Model says:

> Positive

Actual:

> Positive

```text
Model: Disease
Actual: Disease
```

Correct.

---

# 21. True Negative — TN

Model says:

> Negative

Actual:

> Negative

```text
Model: Healthy
Actual: Healthy
```

Correct.

---

# 22. False Positive — FP

Model says:

> Positive

Actual:

> Negative

```text
Model: Disease
Actual: Healthy
```

False alarm.

---

# 23. False Negative — FN

Model says:

> Negative

Actual:

> Positive

```text
Model: Healthy
Actual: Disease
```

🚨 Missed case.

---

# 24. The Easiest Way to Remember

```text id="3qz0cm"
TRUE  → Model was correct
FALSE → Model was wrong

POSITIVE → Model predicted positive
NEGATIVE → Model predicted negative
```

Combine them:

```text
TP → Correct positive
TN → Correct negative
FP → Wrong positive
FN → Wrong negative
```

---

# 25. Precision

Precision asks:

> **"When my model predicts Positive, how often is it actually Positive?"**

Formula:

$$
Precision=\frac{TP}{TP+FP}
$$

Imagine spam detection.

Model says:

```text
100 emails are spam
```

But only:

```text
80 actually are spam
```

Then:

$$
Precision=80\%
$$

---

# 26. Precision Mental Model

> **"Can I trust a positive prediction?"**

High precision:

```text
Predicted Positive
       ↓
Usually actually positive
```

---

# 27. Recall

Recall asks:

> **"Of all the actual positive cases, how many did my model find?"**

Formula:

$$
Recall=\frac{TP}{TP+FN}
$$

Suppose:

```text
100 actual patients have disease
```

Model detects:

```text
90
```

Then:

$$
Recall=90\%
$$

---

# 28. Recall Mental Model

> **"Did I find most of the positives?"**

High recall means:

```text
Actual positives
       ↓
Most detected
```

---

# 29. Precision vs Recall

This is a VERY important distinction.

### Precision

Focuses on:

> **Predicted positives**

### Recall

Focuses on:

> **Actual positives**

Remember:

```text id="gsh8b2"
PRECISION
"Of what I predicted positive,
how many were actually positive?"

RECALL
"Of all actual positives,
how many did I find?"
```

---

# 30. Medical Example

Suppose we're detecting cancer.

A **false negative** is dangerous:

```text
Patient has cancer
       ↓
Model says "No cancer"
```

Therefore:

> **High recall may be very important.**

We want to miss as few actual cases as possible.

---

# 31. Spam Example

Now imagine your email system.

A false positive means:

```text
Important email
       ↓
Marked as spam
```

That's annoying.

So you may care strongly about:

> **High precision**

You don't want legitimate emails incorrectly labeled spam.

---

# 32. Precision-Recall Tradeoff

Often improving one can hurt the other.

Imagine a disease detector.

### Very sensitive model:

```text
Find almost everyone
```

Great recall.

But:

```text
Many healthy people flagged
```

Precision may decrease.

---

# 33. F1 Score

What if we want a balance between precision and recall?

Use:

$$
F1=2\frac{Precision\times Recall}{Precision+Recall}
$$

It's the **harmonic mean**.

Example:

```text
Precision = 0.80
Recall = 0.60
```

Then:

$$
F1 \approx 0.686
$$

F1 is useful when:

> Both precision and recall matter.

---

# 34. Why Not Just Average Them?

Because the harmonic mean strongly penalizes imbalance.

For example:

```text
Precision = 1.0
Recall = 0.1
```

A simple average:

$$
0.55
$$

But F1 is much lower:

$$
F1\approx0.18
$$

That makes sense.

A model shouldn't get a great score merely because one metric is excellent while the other is terrible.

---

# 35. Accuracy vs Precision vs Recall vs F1

| Metric    | Question                                   |
| --------- | ------------------------------------------ |
| Accuracy  | How many predictions were correct overall? |
| Precision | Can I trust positive predictions?          |
| Recall    | Did I find most actual positives?          |
| F1        | How well do precision and recall balance?  |

---

# 36. ROC-AUC

Now we reach another important metric.

ROC-AUC measures how well a classifier can **rank positive examples above negative examples across classification thresholds**.

You don't need to memorize the mathematics yet.

Mental model:

```text id="h5k2e4"
Threshold changes
      ↓
Predictions change
      ↓
Measure classification behavior
      ↓
ROC curve
      ↓
AUC
```

AUC roughly tells us how well the model separates the two classes across thresholds.

---

# 37. Why Threshold Matters

Suppose a model outputs:

```text
Disease probability = 0.73
```

Do we classify as disease?

Depends on threshold.

If:

$$
threshold=0.5
$$

then:

```text
0.73 > 0.5
↓
Positive
```

But if threshold is:

$$
0.8
$$

then:

```text
0.73 < 0.8
↓
Negative
```

The model's score didn't change.

**Our decision threshold changed.**

---

# 38. This Is Extremely Useful

You can often adjust the threshold depending on the business/scientific problem.

For example:

```text
Cancer detection
↓
Lower threshold
↓
Catch more possible cases
↓
Higher recall
```

But potentially:

```text
More false positives
```

---

# 39. Evaluation Is Not Just One Number

A professional ML engineer doesn't simply say:

> "Accuracy = 94%."

Instead:

```text
Accuracy
Precision
Recall
F1
Confusion Matrix
```

and chooses metrics based on the real-world consequences of errors.

---

# 40. Metric Selection Cheat Sheet

```text id="xj9vca"
Regression
   │
   ├── MAE  → typical error
   ├── RMSE → punish large errors
   ├── MSE  → squared error
   └── R²   → relative explanatory performance


Classification
   │
   ├── Accuracy  → balanced classes
   ├── Precision  → avoid false positives
   ├── Recall     → avoid false negatives
   ├── F1         → balance precision/recall
   └── ROC-AUC    → ranking/separation across thresholds
```

---

# 41. The Most Important Question

Don't ask:

> **"Which metric is popular?"**

Ask:

> **"What kind of mistake is expensive?"**

For example:

### Fraud detection

Missing fraud:

```text
FN 🚨
```

may be very costly.

So recall can matter greatly.

### Spam detection

Marking important mail as spam:

```text
FP 🚨
```

may be costly.

So precision can matter greatly.

### House price prediction

You care about:

```text
How many dollars off?
```

MAE/RMSE make sense.

---

# 42. Evaluation Pipeline

Now combine everything:

```text id="5i7gci"
              Trained Model
                    │
                    ▼
              Unseen Data
                    │
                    ▼
               Predictions
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      Regression         Classification
          │                   │
          ↓                   ↓
    MAE / RMSE / R²     Confusion Matrix
                              │
                     ┌────────┼────────┐
                     ↓        ↓        ↓
                 Precision  Recall     F1
```

---

# 43. One Critical Rule 🚨

### Never evaluate on the training data alone.

Suppose:

```text
Train accuracy = 99%
```

That doesn't tell us enough.

We need:

```text
Unseen validation/test performance
```

because that's what tells us whether the model generalizes.

---

# 44. The Complete ML Lifecycle So Far

You should now have this mental map:

```text id="9m8d1e"
1. Problem Definition
          ↓
2. Data Collection
          ↓
3. EDA
          ↓
4. Cleaning
          ↓
5. Feature Engineering
          ↓
6. Model Selection
          ↓
7. Training
          ↓
8. Validation
          ↓
9. Evaluation
          ↓
10. Deployment
          ↓
11. Monitoring
          ↓
12. Retraining ↺
```

---

# 🧠 Final Memory Trick

Imagine you're hiring an employee.

### Training

You teach them.

### Validation

You give practice tasks.

### Testing

You give the final exam.

### Evaluation

You inspect their performance.

### Metrics

You decide **what "good performance" actually means**.

And that's the key:

> **A model is not "good" in isolation. It is good relative to the metric and real-world objective that matter.**

---

## 🚀 Next: Part 9 — Model Improvement & Hyperparameter Tuning

Now we know our model isn't performing perfectly.

So what do we do?

We'll learn how to systematically improve it:

```text
Poor Performance
      ↓
Diagnose the problem
      ↓
Underfitting or Overfitting?
      ↓
Better features?
More data?
Different model?
Hyperparameter tuning?
Regularization?
      ↓
Retrain
      ↓
Evaluate again
```

And we'll finally connect **bias, variance, regularization, hyperparameters, GridSearchCV, RandomizedSearchCV, and pipelines** into one practical workflow.
