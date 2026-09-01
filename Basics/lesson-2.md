# 📚 ML Foundation — Lesson 2

# Dataset → Samples → Features → Labels → Targets

Now that you understand **AI vs ML vs DL** and the **types of ML**, the next thing we need to master is the language ML uses to describe data.

These five words appear **everywhere**:

> **Dataset, Sample, Feature, Label, Target**

If these become second nature, you'll understand almost every ML tutorial much more easily.

---

# 1. 🗃️ What is a Dataset?

A **dataset** is simply a collection of data used for analysis or machine learning.

Imagine we want to predict house prices.

We collect information about many houses:

| House | Size | Bedrooms | Age | Location  | Price |
| ----- | ---: | -------: | --: | --------- | ----: |
| 1     | 1000 |        2 |  10 | Lahore    |   15M |
| 2     | 1500 |        3 |   5 | Lahore    |   22M |
| 3     | 2000 |        4 |   3 | Islamabad |   35M |
| 4     | 1200 |        2 |   8 | Karachi   |   18M |
| 5     | 2500 |        5 |   2 | Islamabad |   45M |

This entire table is our:

> **Dataset**

Think of a dataset as a **book of examples** that we're giving to our ML algorithm.

---

# 2. 📄 What is a Sample?

A **sample** is one individual observation/example inside the dataset.

Look at this row:

| Size | Bedrooms | Age | Location | Price |
| ---: | -------: | --: | -------- | ----: |
| 1500 |        3 |   5 | Lahore   |   22M |

This represents **one house**.

Therefore:

> One house = one sample.

If our dataset contains information about 10,000 houses:

$$
\boxed{\text{10,000 samples}}
$$

### Analogy

Imagine a class containing 100 students.

```text
Class
 │
 ├── Student 1
 ├── Student 2
 ├── Student 3
 ├── ...
 └── Student 100
```

The **class = dataset**.

Each **student = sample**.

---

# 3. 🧩 What is a Feature?

Now look at what describes each house:

```text
Size
Bedrooms
Age
Location
```

These are **features**.

A feature is an input variable that provides information to the ML model.

For our house:

```text
Size = 1500 sq ft
Bedrooms = 3
Age = 5 years
Location = Lahore
```

These pieces of information help the model predict the price.

So:

> **Feature = information/input used by the model to make a prediction.**

---

# 4. 🎒 Student Example

Suppose we want to predict whether a student will pass.

Our dataset:

| Study Hours | Attendance | Assignments | Sleep | Result |
| ----------: | ---------: | ----------: | ----: | ------ |
|           2 |        60% |         40% |    5h | Fail   |
|           4 |        75% |         70% |    7h | Pass   |
|           1 |        50% |         30% |    6h | Fail   |
|           6 |        90% |         95% |    8h | Pass   |

Features are:

```text
Study Hours
Attendance
Assignments
Sleep
```

because these are the pieces of information we're giving the model.

---

# 5. 🎯 What is the Target?

Now ask:

> **What are we trying to predict?**

In our student example:

```text
Pass / Fail
```

That's our **target**.

In our house example:

```text
Price
```

is our target.

Therefore:

> **Target = the value our model is trying to predict.**

Think:

```text
Features ──────────→ Model ──────────→ Target
   X                                      y
```

For example:

```text
Study hours ─────┐
Attendance ──────┤
Assignments ─────┤──→ ML Model ──→ Pass/Fail
Sleep ───────────┘
```

---

# 6. 🏷️ What is a Label?

This one causes confusion.

In many ML contexts, **label** means the known correct answer associated with a training example.

For example:

| Image | Label |
| ----- | ----- |
| 🐱    | Cat   |
| 🐕    | Dog   |
| 🐱    | Cat   |
| 🐕    | Dog   |

The labels are:

```text
Cat
Dog
Cat
Dog
```

The model uses these known answers during supervised training.

So:

> **Label = the known answer/class associated with an example.**

---

# 7. Target vs Label

These terms overlap, but they're not always interchangeable.

### Classification

Suppose:

```text
Email → Spam
```

"Spam" can be called:

* label
* target
* target value
* class

### Regression

Suppose:

```text
House → $25 million
```

We normally call `$25M` the:

> **target**

rather than a label.

A useful beginner rule:

```text
Classification → label / target
Regression     → target
```

Don't get too obsessed with the terminology yet. Different libraries, books, and practitioners sometimes use these words slightly differently.

---

# 8. 🧠 The Most Important ML Picture

Memorize this:

```text
                    DATASET
                       │
        ┌──────────────┴──────────────┐
        │                             │
     FEATURES                       TARGET
       X                               y
        │                              │
        │                              │
        └──────────→ MODEL ←───────────┘
                       │
                    Learning
                       │
                       ↓
                  Prediction
```

During supervised learning, we give the model both **X and y** so it can learn their relationship.

Later, for a new example, we give it only **X**:

```text
New X
 ↓
Model
 ↓
Prediction ŷ
```

---

# 9. 🔥 Training vs Prediction

This distinction is extremely important.

Suppose we're training a house-price model.

### Training

We have:

```text
Size
Bedrooms
Age
Location
   +
Actual Price
```

The model sees the correct answers.

```text
X + y
 ↓
Training
 ↓
Model
```

---

### Prediction

Now we find a new house.

```text
Size = 1800
Bedrooms = 3
Age = 4
Location = Lahore
```

We don't know its actual price yet.

So:

```text
X
 ↓
Trained Model
 ↓
Predicted Price
```

The model produces:

$$
\hat{y}
$$

Read this as:

> **y-hat = predicted y**

---

# 10. 🧪 A Real ML Example

Let's make this concrete.

Suppose you're building a food-delivery ETA predictor.

Your data might look like:

| Distance | Traffic | Restaurant Prep Time | Weather | Delivery Time |
| -------: | ------: | -------------------: | ------- | ------------: |
|     2 km |     Low |               10 min | Clear   |        20 min |
|     5 km |    High |               15 min | Rain    |        45 min |
|     3 km |  Medium |               10 min | Clear   |        28 min |
|     8 km |    High |               20 min | Rain    |        60 min |

### Dataset

Entire table.

### Samples

Each delivery.

### Features

```text
Distance
Traffic
Restaurant Prep Time
Weather
```

### Target

```text
Delivery Time
```

So:

$$
X =
\begin{bmatrix}
distance & traffic & prep & weather
\end{bmatrix}
$$

and:

$$
y = delivery\ time
$$

---

# 11. 🚕 Your Feature Engineering Connection

This connects directly with something you'll learn later.

Suppose you're predicting taxi fare.

Raw features:

```text
Distance
Traffic
Passenger count
Time
Weather
```

You might create a new feature:

```text
Traffic-adjusted distance
```

or perhaps:

```text
Distance / average speed
```

These are also features.

This process is called:

> **Feature Engineering**

We'll study it later in depth.

---

# 12. 📊 What is X and y?

You'll see this constantly in Python/Scikit-learn.

Usually:

$$
X = \text{features}
$$

$$
y = \text{target}
$$

For example:

```python
X = df[["study_hours", "attendance", "assignments"]]

y = df["result"]
```

Think:

```text
X = Questions / Evidence
y = Answer
```

---

# 13. 👨‍🏫 Teacher Analogy

Imagine an ML teacher.

You give the teacher:

```text
Student information → Actual result

2 hours, 60%, 40% → Fail
5 hours, 85%, 90% → Pass
6 hours, 90%, 95% → Pass
```

The teacher studies the relationship.

Then you introduce:

```text
New student:
4 hours, 80%, 75%
```

and ask:

> "What do you predict?"

The model says:

```text
Pass
```

This is supervised learning.

---

# 14. 🧠 A Crucial Distinction: Feature ≠ Sample

Beginners often mix these up.

Suppose:

| Size | Bedrooms | Price |
| ---: | -------: | ----: |
| 1000 |        2 |   15M |
| 1500 |        3 |   22M |
| 2000 |        4 |   35M |

There are:

**3 samples**

and:

**2 input features**

```text
Samples = rows
Features = columns describing the input
```

Usually, when working with tabular ML data:

$$
X.shape = (n\_samples,\ n\_features)
$$

For our example:

$$
X.shape=(3,2)
$$

That means:

```text
3 rows
2 feature columns
```

This is going to become extremely important when you work with **NumPy**, which you've already been learning.

---

# 15. 🔢 NumPy Connection

Suppose:

```python
import numpy as np

X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4]
])

y = np.array([
    15,
    22,
    35
])
```

Then:

```python
X.shape
```

gives:

```text
(3, 2)
```

Meaning:

```text
3 samples
2 features
```

And:

```python
y.shape
```

gives:

```text
(3,)
```

because there is one target value for each sample.

---

# 16. 🖼️ But What About Images?

Here's where things become really interesting for your computer-vision path.

Suppose you have:

```text
10,000 images of cats and dogs
```

Each image is a sample.

```text
Dataset
 │
 ├── Image 1 → Cat
 ├── Image 2 → Dog
 ├── Image 3 → Cat
 ├── Image 4 → Dog
 └── ...
```

So:

> **Each image = one sample.**

The pixels contain the information from which the model learns.

For a simple grayscale image:

```text
28 × 28 pixels
```

you could think of it as:

$$
28\times28=784
$$

pixel values.

So one image can be represented as **784 numerical features** in a flattened representation.

Deep learning, however, can work directly with the image's spatial structure rather than requiring you to manually flatten it for every model.

---

# 17. 🧬 What About Text?

Suppose you're classifying messages:

```text
"Congratulations! You won $1000!"
```

One message = one sample.

The target:

```text
Spam
```

But computers cannot directly feed raw English sentences into most traditional ML algorithms.

We need to convert text into numerical representations.

For example:

```text
Text
 ↓
Numerical representation
 ↓
Features
 ↓
ML model
 ↓
Prediction
```

This conversion is part of **feature representation / feature engineering**, and modern deep learning uses powerful learned representations such as embeddings.

---

# 18. 🌎 One Concept, Many Domains

| Problem            | Sample        | Features              | Target              |
| ------------------ | ------------- | --------------------- | ------------------- |
| House price        | House         | Size, rooms, location | Price               |
| Student prediction | Student       | Study, attendance     | Pass/Fail           |
| Spam detection     | Email         | Text characteristics  | Spam/Not Spam       |
| Medical diagnosis  | Patient/image | Measurements/pixels   | Diagnosis           |
| Taxi prediction    | Trip          | Distance, traffic     | Fare/ETA            |
| Object detection   | Image         | Image pixels          | Objects + locations |

The **structure remains the same**.

That's the beautiful thing about ML.

---

# 🧠 19. The Ultimate Mental Model

Whenever you encounter a new ML problem, ask these four questions:

### Question 1

> **What is one sample?**

Example:

```text
One house
One student
One email
One image
One transaction
One delivery
```

### Question 2

> **What information do I have about it?**

Those are your:

**Features (X)**

### Question 3

> **What am I trying to predict?**

That's your:

**Target (y)**

### Question 4

> **Do I already know the correct target during training?**

If yes:

> **Supervised learning**

If no, perhaps:

> **Unsupervised learning**

And if the system learns through actions and rewards:

> **Reinforcement learning**

---

# 🧠 Your "Never Forget" Formula

Remember this sentence:

> **A dataset contains samples. Each sample has features. In supervised learning, the sample also has a known target/label. The model learns the relationship between X and y so it can predict y for new X.**

Or even shorter:

```text
DATASET
   ↓
SAMPLES
   ↓
FEATURES (X) ──→ MODEL ──→ TARGET (y)
                         ↑
                    Prediction
```

And during training:

```text
X + y
 ↓
Learn
 ↓
MODEL
```

During prediction:

```text
New X
 ↓
MODEL
 ↓
ŷ
```

---

## 🚀 Next Lesson

Now that we know **what the data looks like**, the natural next question is:

> **"How do we actually teach the model using that data?"**

That leads us to:

### **Training Set vs Validation Set vs Test Set**

We'll use a **student-exam analogy**, then build the actual ML workflow:

```text
Raw Dataset
     ↓
Train / Validation / Test
     ↓
Training
     ↓
Evaluation
     ↓
Final Model
```

This is where **overfitting** starts to become much clearer.
