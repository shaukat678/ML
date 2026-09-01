# 📚 ML Foundation — Lesson 3

# Training Set vs Validation Set vs Test Set

Now we reach one of the **most important ideas in practical Machine Learning**.

You already know:

```text
Dataset
   ↓
Samples
   ↓
Features (X)
Target (y)
```

But here's the problem:

> **How do we know whether our model has actually learned, rather than simply memorized the training data?**

That's where we split our data.

---

# 🎓 1. The Best Analogy: Student Preparing for an Exam

Imagine you're preparing for a university exam.

You have a huge book containing **1,000 questions**.

You divide them into three groups:

```text
1000 questions
      │
      ├───────────────┐
      ↓               ↓
   Practice        Questions
   Questions       you've never
                    seen before
```

But in ML we usually make **three roles**:

```text
                 DATASET
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      TRAIN       VALIDATION    TEST
        │             │           │
     Learn          Tune       Final exam
```

Think of them as:

| ML             | Student analogy    |
| -------------- | ------------------ |
| Training set   | Practice questions |
| Validation set | Mock exam          |
| Test set       | Final exam         |

This analogy is worth remembering.

---

# 🧑‍🏫 2. Training Set

The **training set** is the data the model actually learns from.

Suppose we have:

```text
10,000 houses
```

We might use:

```text
8,000 → Training
```

The model sees:

```text
Features + actual target
```

For example:

```text
1500 sq ft
3 bedrooms
Lahore
5 years old

Actual price = 22M
```

The model makes a prediction.

Perhaps:

```text
Prediction = 19M
```

There's an error:

$$
22M - 19M = 3M
$$

The model uses this error to adjust itself.

This happens repeatedly.

```text
Training data
     ↓
Prediction
     ↓
Error
     ↓
Adjust model
     ↓
Prediction improves
     ↓
Repeat
```

This is **training**.

---

# 🎯 3. Validation Set

Now suppose we've trained a model.

But we want to make decisions such as:

> Which model should I use?

Maybe we have:

```text
Model A → Linear Regression
Model B → Decision Tree
Model C → Random Forest
Model D → Neural Network
```

Which one is better?

We need another set of data that wasn't used to directly fit the model.

That's the:

> **Validation set**

For example:

```text
8,000 → Training
1,000 → Validation
1,000 → Test
```

We train the models on the 8,000 training examples.

Then compare them on the validation set.

```text
                 Validation
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    Model A        Model B       Model C
       ↓             ↓             ↓
     Score          Score         Score
```

Maybe:

```text
Linear Regression → RMSE = 5.2
Decision Tree     → RMSE = 4.1
Random Forest     → RMSE = 3.2
```

We might choose Random Forest.

---

# 🧪 4. Test Set

Now comes the **final exam**.

The test set should ideally remain untouched until you're ready for the final evaluation.

Suppose:

```text
8,000 → Training
1,000 → Validation
1,000 → Test
```

You use:

```text
Training → learn
Validation → make choices
Test → final evaluation
```

You evaluate the final selected model on the test set.

If it performs well there, you have stronger evidence that it generalizes to unseen data.

---

# 🧠 5. Why Can't We Just Train and Test on the Same Data?

This is one of the most important ideas in ML.

Imagine you give a student the exact exam questions beforehand:

```text
Question 1
Question 2
Question 3
...
```

The student memorizes the answers.

Then you give the exact same questions.

Result:

> 100% marks! 🎉

Did the student actually understand the subject?

Not necessarily.

They may have simply memorized the answers.

The same thing can happen to an ML model.

---

# 🤖 6. ML Version of Memorization

Suppose we train a model on:

```text
House A → 15M
House B → 22M
House C → 35M
...
```

The model performs incredibly well on these exact houses.

```text
Training accuracy = 99.9%
```

Sounds amazing.

But then we give it new houses:

```text
House X
House Y
House Z
```

and performance collapses.

That's a warning sign.

The model may have:

# Overfit

---

# 🔥 7. What is Overfitting?

Overfitting means:

> **The model learns the training data too specifically and fails to generalize well to new unseen data.**

Imagine a student who memorizes:

```text
Question 1 → Answer A
Question 2 → Answer C
Question 3 → Answer B
```

but doesn't understand the underlying concepts.

Change the wording of the question:

> Student: 😵

That's overfitting.

---

# 🌳 8. A Real ML Example

Imagine we're predicting whether students will pass.

Training data:

| Study | Attendance | Result |
| ----: | ---------: | ------ |
|    2h |        60% | Fail   |
|    3h |        65% | Fail   |
|    5h |        80% | Pass   |
|    6h |        90% | Pass   |

Suppose our model becomes overly complicated and effectively memorizes these students.

Training:

```text
Accuracy = 100%
```

But new students:

| Study | Attendance | Actual |
| ----: | ---------: | ------ |
|    4h |        78% | Pass   |
|    5h |        72% | Pass   |
|  2.5h |        68% | Fail   |

The model might perform badly.

So:

```text
Training performance ↑
Test performance ↓
```

That's a classic overfitting pattern.

---

# 🌱 9. Underfitting

There's an opposite problem.

Suppose our model is **too simple**.

It doesn't capture the important patterns even in the training data.

For example:

```text
Training performance → poor
Test performance     → poor
```

That's:

# Underfitting

Think of a student who barely studied anything.

```text
Practice exam → 40%
Final exam    → 42%
```

The student didn't even learn the training material properly.

---

# ⚖️ 10. Overfitting vs Underfitting

|                      | Underfitting | Good fit             | Overfitting       |
| -------------------- | ------------ | -------------------- | ----------------- |
| Training performance | Poor         | Good                 | Excellent         |
| Test performance     | Poor         | Good                 | Poor              |
| Model                | Too simple   | Appropriate          | Too complex       |
| Analogy              | Didn't study | Understands concepts | Memorized answers |

The goal:

> **Learn enough to capture the real pattern, but not so much that you memorize the training data.**

---

# 📈 11. The Model's Real Goal: Generalization

This is one of the most important words in ML.

### Generalization

> **The ability of a model to perform well on new, unseen data.**

Imagine:

```text
Training data
     ↓
Learn underlying pattern
     ↓
New data
     ↓
Good prediction
```

That's what we want.

Not:

```text
Training data
     ↓
Memorize
     ↓
New data
     ↓
😵
```

---

# 🧪 12. Why Three Sets?

Now the roles should become crystal clear.

### Training set

> **"Learn from this."**

### Validation set

> **"Use this to make decisions."**

### Test set

> **"Now show me how well your final system really performs."**

A useful analogy:

```text
Training     = Practice
Validation   = Mock exam
Test         = Final exam
```

---

# 🔧 13. What Decisions Do We Make Using Validation Data?

Suppose you're building a model.

You could use:

```text
Decision Tree
Random Forest
SVM
Neural Network
```

And each has settings called:

> **Hyperparameters**

For example, a decision tree has settings such as maximum depth.

You might test:

```text
depth = 2
depth = 5
depth = 10
depth = 20
```

You train each configuration and compare them using the validation set.

Maybe:

| Tree depth | Validation RMSE |
| ---------: | --------------: |
|          2 |             8.5 |
|          5 |             5.2 |
|         10 |             4.1 |
|         20 |             6.8 |

Interesting.

Depth 20 performs worse.

Why?

It may be becoming too complex and overfitting.

So we might choose:

```text
depth = 10
```

based on validation performance.

---

# ⚠️ 14. Why Not Keep Looking at the Test Set?

This is subtle but extremely important.

Imagine you repeatedly evaluate your model on the test set:

```text
Test → Model A
Test → Model B
Test → Model C
Test → Model D
Test → Model E
...
```

Eventually you choose the model that happens to perform best on that particular test set.

Now you've indirectly used the test set to make decisions.

The test set is no longer a truly independent final exam.

So ideally:

> **Don't use the test set to repeatedly tune your model.**

Use validation data for those decisions.

---

# 🔄 15. The Real ML Workflow

Now we can build a much more realistic picture:

```text
                 RAW DATA
                    │
                    ↓
              Clean / Prepare
                    │
                    ↓
                 Split
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
    TRAIN        VALIDATION      TEST
       │            │            │
       ↓            │            │
    Train           │            │
       │            │            │
       └──────→ Evaluate ←───────┘
                    │
                    ↓
              Choose model
                    │
                    ↓
             Final evaluation
                    │
                    ↓
                  TEST
```

A more precise workflow is:

```text
Train
  ↓
Validation
  ↓
Tune model
  ↓
Train again / finalize
  ↓
Test ONCE at the end
```

---

# 🔢 16. Typical Splits

There isn't one universal split.

Common examples include:

```text
80% Train
20% Test
```

or:

```text
70% Train
15% Validation
15% Test
```

or:

```text
80% Train
10% Validation
10% Test
```

The exact split depends on:

* dataset size
* problem
* model
* evaluation strategy

For very small datasets, we often use **cross-validation**, which we'll learn later.

---

# 🧠 17. What Happens When Dataset Is Tiny?

Suppose you only have:

```text
100 samples
```

If you do:

```text
80 train
10 validation
10 test
```

that's not a lot of data for learning or reliable evaluation.

Instead, we might use:

# Cross-Validation

For example:

```text
100 samples

Fold 1:
80 train + 20 validation

Fold 2:
80 train + 20 validation

Fold 3:
80 train + 20 validation

...
```

Each sample can participate in validation across different folds.

This gives a more robust estimate of model performance.

We'll study this properly after understanding the basic workflow.

---

# 🚨 18. A Huge Practical Problem: Data Leakage

Now you're ready for one of the most important practical ML concepts.

Imagine you're predicting whether a patient has a disease.

You accidentally include:

```text
"Final diagnosis"
```

as one of your input features.

The model sees:

```text
Patient information
+
Final diagnosis
        ↓
      Model
        ↓
     Disease
```

Of course it performs amazingly!

But this isn't real learning.

The answer leaked into the input.

That's:

# Data Leakage

Another example:

You're predicting tomorrow's stock price but accidentally use information from tomorrow.

The model appears brilliant.

But it's cheating.

---

# 🧠 19. The Golden Rule

When splitting your data:

> **Information from the future or evaluation data should not leak into the training process.**

This becomes especially important when you later learn:

* preprocessing
* normalization
* standardization
* feature engineering
* target encoding
* time-series ML

For example, suppose you want to standardize your data.

You should generally:

```text
Training data
     ↓
Fit scaler
     ↓
Transform training
     ↓
Use SAME scaler
     ↓
Transform validation/test
```

Not:

```text
Entire dataset
     ↓
Fit scaler
```

because that can allow information from the validation/test sets to influence preprocessing.

We'll revisit this when we study **data preprocessing**.

---

# 🧪 20. Let's Simulate an ML Project

Imagine we're building:

> 🚕 **Taxi Fare Prediction**

We have:

```text
100,000 trips
```

Each sample contains:

```text
Distance
Traffic
Passenger count
Pickup time
Weather
Car type
```

Target:

```text
Fare
```

We split:

```text
80,000 → Training
10,000 → Validation
10,000 → Test
```

### Step 1

Train several models:

```text
Linear Regression
Decision Tree
Random Forest
Gradient Boosting
```

### Step 2

Evaluate them on validation data.

```text
Linear Regression → RMSE 8.2
Decision Tree     → RMSE 6.5
Random Forest     → RMSE 4.9
Gradient Boosting → RMSE 4.4
```

Choose Gradient Boosting.

### Step 3

Finalize the model.

### Step 4

Evaluate on the untouched test set.

```text
Test RMSE = 4.7
```

Now we have a much more realistic estimate of how the final model behaves on unseen data.

---

# 🧠 21. One Important Detail: Training Error vs Validation Error

Imagine we're training a model and progressively increasing its complexity.

You might see something like:

```text
Model complexity
        →
        
Training error:
   ↓ ↓ ↓ ↓ ↓

Validation error:
   ↓ ↓ ↓ ↑ ↑
```

Initially:

```text
Model too simple
→ high training error
→ high validation error
```

Then:

```text
Model becomes appropriate
→ training error decreases
→ validation error decreases
```

Eventually:

```text
Model becomes too complex
→ training error keeps decreasing
→ validation error starts increasing
```

That turning point is a classic sign of **overfitting**.

---

# 🎯 22. The Three Questions You Should Always Ask

Whenever you build an ML model:

### Question 1

> **Did the model learn the training data?**

Check training performance.

### Question 2

> **Does it generalize to data it wasn't trained on?**

Check validation/test performance.

### Question 3

> **Did I accidentally let information from outside the training process leak into the model?**

Check for data leakage.

These three questions will save you from a huge number of ML mistakes.

---

# 🧠 23. Your Mental Map So Far

You now have:

```text
                         MACHINE LEARNING
                                │
                                ↓
                              DATA
                                │
                                ↓
                         ┌─────────────┐
                         │   Dataset   │
                         └─────────────┘
                                │
                         contains samples
                                │
                                ↓
                      ┌──────────────────┐
                      │ Sample           │
                      │ Features → X     │
                      │ Target → y       │
                      └──────────────────┘
                                │
                                ↓
                             Split
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
           TRAIN            VALIDATION           TEST
              │                 │                 │
              ↓                 ↓                 ↓
            Learn          Tune / choose       Final
                                              evaluation
              │
              ↓
             MODEL
              │
              ↓
         New unseen data
              │
              ↓
         Generalization
```

---

# 🔥 The One Sentence I Want You to Remember

> **Training data teaches the model, validation data helps us choose and tune the model, and test data gives us the final unbiased-ish check of how well the finished model generalizes to unseen data.**

And:

> **A model that performs brilliantly on training data but poorly on unseen data is probably overfitting.**

---

# 🚀 Next Lesson

Now we have the **data → split → training → evaluation** picture.

The next question is:

> **What exactly is happening INSIDE the model when it learns?**

We'll enter the heart of ML:

### **Model → Parameters → Weights → Predictions → Loss Function → Optimization**

We'll use **house-price prediction and a simple line** to understand:

$$
y = wx+b
$$

Then we'll gradually arrive at:

$$
\boxed{\text{Gradient Descent}}
$$

This is the bridge between "I know what ML is" and **"I understand how ML actually learns."**
