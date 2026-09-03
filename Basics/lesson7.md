# 📚 ML Foundation — Lesson 7

# Problem Definition & Data Collection

### From a Real-World Problem → An ML Problem

Now we start using the **first two stages of the ML lifecycle properly**.

The goal is not just to memorize steps. I want you to develop the habit of thinking like an **ML engineer**.

---

# 🧠 First: Where Are We in the ML Lifecycle?

Keep this diagram in mind:

```text
┌─────────────────────────────────────────────────────────────┐
│              MACHINE LEARNING DEVELOPMENT LIFECYCLE         │
└─────────────────────────────────────────────────────────────┘

       ┌──────────────────┐
       │ 1. PROBLEM       │
       │    DEFINITION    │
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ 2. DATA          │
       │    COLLECTION    │
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ 3. DATA          │
       │    UNDERSTANDING  │
       │    / EDA          │
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ 4. DATA          │
       │    PREPARATION   │
       └────────┬─────────┘
                ↓
              ...
```

Today we're focusing on:

> **Problem Definition → Data Collection**

---

# 🎯 PART 1 — What Is Problem Definition?

Before writing:

```python
import sklearn
```

you need to answer:

> **What exactly are we asking the machine to predict or decide?**

This sounds simple.

But it's one of the most important steps in an ML project.

---

# 🏠 Real-World Scenario

Imagine a real-estate company approaches you.

They say:

> "We want to use AI to predict house prices."

That's **not yet an ML problem definition**.

It's just an idea.

You need to convert it into something precise.

---

# 🔄 Business Problem → ML Problem

```text
                 REAL WORLD
                     │
                     ↓
          "We want better pricing"
                     │
                     ↓
             Define objective
                     │
                     ↓
       ┌─────────────────────────┐
       │ Predict house price      │
       │ for a new property       │
       └────────────┬────────────┘
                    ↓
             Define inputs
                    │
                    ↓
       Size, location, bedrooms,
       age, condition, etc.
                    │
                    ↓
             Define target
                    │
                    ↓
              House Price
                    │
                    ↓
            Choose ML problem
                    │
                    ↓
               REGRESSION
```

That's the transformation we want.

---

# 🧩 1. Define the Objective

Start with a sentence:

> **Given information about a house, predict its selling price.**

Now we have something concrete.

Mathematically:

$$
X \rightarrow y
$$

where:

$$
X = \text{house information}
$$

and:

$$
y = \text{house price}
$$

---

# 🔢 2. Identify the Features

Features are the information available to the model.

For example:

```text
┌─────────────────────────────┐
│          HOUSE              │
├─────────────────────────────┤
│ Size          → 1800 sq ft  │
│ Bedrooms      → 3           │
│ Bathrooms     → 2           │
│ Location      → DHA         │
│ Age           → 5 years     │
│ Condition     → Good        │
└─────────────────────────────┘
```

These are potential:

$$
\boxed{\text{Features } X}
$$

---

# 🎯 3. Identify the Target

The target is what we want the model to predict.

Here:

```text
Features
   ↓
┌────────────────────┐
│ Size               │
│ Bedrooms           │
│ Location           │
│ Age                │
│ Condition          │
└─────────┬──────────┘
          ↓
        MODEL
          ↓
┌────────────────────┐
│ Predicted Price    │
└────────────────────┘
```

So:

$$
\boxed{y=\text{house price}}
$$

---

# 🔥 Features vs Target

Memorize this:

```text
                 DATASET
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
       FEATURES             TARGET
          X                   y
          │                   │
    Information          What we want
     we give model        model to predict
```

Example:

| Size | Bedrooms | Location | Age | Price |
| ---: | -------: | -------- | --: | ----: |
| 1000 |        2 | A        |  10 |   15M |
| 1500 |        3 | B        |   5 |   25M |
| 2000 |        4 | A        |   3 |   35M |

Here:

$$
X =
\begin{bmatrix}
Size & Bedrooms & Location & Age
\end{bmatrix}
$$

and:

$$
y=Price
$$

---

# 🧠 4. Decide What Type of ML Problem It Is

Now ask:

> **What kind of output do I want?**

This determines the ML problem type.

---

## 🟢 Case A — Numerical Prediction

Example:

> Predict house price.

Output:

```text
25,000,000
```

That's:

# Regression

```text
Input
  ↓
ML
  ↓
Number
```

Examples:

```text
House price
Temperature
Salary
Taxi fare
Sales
Demand
```

---

# 🔵 Case B — Category Prediction

Suppose instead we ask:

> "Will this customer default on the loan?"

Output:

```text
Yes / No
```

That's:

# Classification

```text
Input
  ↓
ML
  ↓
Category
```

Examples:

```text
Spam / Not Spam
Cat / Dog
Fraud / Legitimate
Disease / No Disease
Pass / Fail
```

---

# 🟣 Case C — Grouping Without Labels

Suppose you have customers but don't know their categories.

You want the algorithm to discover groups:

```text
Customer data
      ↓
     ML
      ↓
┌─────┬─────┬─────┐
│ G1  │ G2  │ G3  │
└─────┴─────┴─────┘
```

This is:

# Clustering

A type of **unsupervised learning**.

---

# 🟠 Case D — Decision Through Interaction

Suppose an AI controls a robot.

```text
Environment
     ↓
   Robot
     ↓
   Action
     ↓
Reward / Penalty
     ↓
Learn
     ↺
```

That's:

# Reinforcement Learning

---

# 🧠 Problem-Type Decision Tree

Use this mental diagram:

```text
                    ML PROBLEM
                        │
             ┌──────────┼──────────┐
             ↓          ↓          ↓
          Labeled?    No labels?  Interaction?
             │          │          │
        ┌────┴────┐     ↓          ↓
        ↓         ↓  Unsupervised Reinforcement
    Number?    Category?
        │         │
        ↓         ↓
   Regression Classification
```

This decision should happen **before choosing an algorithm**.

---

# 🚨 5. A Very Important Question:

# "Can ML Even Solve This?"

Not every problem requires ML.

Suppose someone asks:

> "Convert Celsius to Fahrenheit."

We already know:

$$
F=\frac95C+32
$$

Why train ML?

You don't need it.

A normal formula is better.

---

# 🧮 Traditional Programming vs ML

Traditional:

```text
             RULES
               +
              DATA
               ↓
            PROGRAM
               ↓
             OUTPUT
```

ML:

```text
             DATA
               +
            OUTPUTS
               ↓
        ML ALGORITHM
               ↓
             MODEL
               ↓
        New Input → Prediction
```

This is a fundamental difference.

---

# 🧠 When Is ML Useful?

ML becomes attractive when:

### Rules are difficult to write manually.

Example:

> Recognize whether an image contains a cat.

You could try to manually write:

```text
IF ears look like this
AND eyes look like this
AND fur looks like this
...
```

That quickly becomes impractical.

ML can instead learn patterns from many examples.

---

# 🚗 Another Example

Traffic prediction.

Could you manually write:

```text
IF Monday
AND 8:00 AM
AND rain
AND road X
AND school open
...
```

Real-world interactions become extremely complicated.

ML can learn patterns from historical data.

---

# 🎯 6. Define the Prediction Time

This is a **professional-level habit**.

Ask:

> **At what moment will the model make the prediction?**

Suppose you're predicting taxi fare.

If prediction happens:

### Before trip

Available:

```text
Distance estimate
Pickup location
Destination
Time
Weather
Traffic
```

But not:

```text
Final trip duration
Final fare
Payment method
```

because they don't exist yet.

This leads to a crucial rule:

# 🚨 Only use information available at prediction time.

Otherwise:

> **Data leakage**

---

# 🧠 Reference Diagram — Prediction Boundary

```text
                    TIME
──────────────────────────────────────────────→

        BEFORE TRIP             AFTER TRIP
             │                       │
             │                       │
      Available features       Future information
             │                       │
             ↓                       ↓
       ┌─────────────┐          ┌──────────────┐
       │    MODEL    │          │ Final fare   │
       │             │          │ Final time   │
       └──────┬──────┘          │ Payment      │
              │                 └──────────────┘
              ↓
       PREDICT FARE

             ↑
             │
       Prediction point
```

The model should only use the left side.

---

# 📦 PART 2 — Data Collection

Once the problem is defined:

> **Where will the data come from?**

This is Stage 2.

---

# 🌍 7. Sources of ML Data

There are many possibilities.

```text
                    DATA
                     │
     ┌───────────────┼────────────────┐
     ↓               ↓                ↓
 Databases        APIs             Sensors
     │               │                │
     ↓               ↓                ↓
 Companies       Weather API      IoT devices
 Hospitals       Maps API         Cameras
 Banks           Financial API    Machines
```

Other sources:

```text
Websites
Surveys
Experiments
Public datasets
Logs
Mobile apps
Satellites
Social media
Scientific instruments
```

---

# ⚛️ 8. Physics Example

Imagine you're studying quantum noise.

Your data could come from:

```text
┌───────────────────────────┐
│ Quantum Experiment        │
├───────────────────────────┤
│ Circuit configuration     │
│ Measurement outcomes      │
│ Expectation values        │
│ Noise parameters          │
│ Time                      │
│ Qubit information         │
└──────────────┬────────────┘
               ↓
             Dataset
```

This is exactly the same ML lifecycle.

The only difference is the **domain**.

---

# 📸 9. Computer Vision Example

Suppose we're building:

> "Detect cars in traffic-camera footage."

We need:

```text
Camera
  ↓
Video
  ↓
Frames
  ↓
Images
  ↓
Annotations
  ↓
ML Dataset
```

For object detection, annotations might look like:

```text
Image
 ┌──────────────────────────────┐
 │                              │
 │     ┌─────────┐              │
 │     │   🚗    │              │
 │     └─────────┘              │
 │                    ┌─────┐   │
 │                    │ 🚗  │   │
 │                    └─────┘   │
 │                              │
 └──────────────────────────────┘
```

Each bounding box needs a label.

---

# 🏷️ 10. Labeled vs Unlabeled Data

This is critical.

### Labeled:

```text
Image → Cat
Image → Dog
Image → Cat
```

We know the answer.

### Unlabeled:

```text
Image
Image
Image
Image
```

No known target.

---

# 🧠 Supervised Learning

```text
Input X + Correct answer y
             ↓
           MODEL
             ↓
       Learn relationship
```

Example:

```text
Study Hours → Exam Score
```

---

# 🧠 Unsupervised Learning

```text
Input X
  ↓
MODEL
  ↓
Discover structure
```

Example:

```text
Customer data
  ↓
Groups
```

---

# 📊 11. How Much Data Do We Need?

There is no universal number.

It depends on:

```text
Problem complexity
Number of features
Noise
Model complexity
Data diversity
Desired accuracy
```

For example:

### Simple problem

```text
Linear relationship
Small dataset
```

may work with relatively little data.

### Complex image problem

```text
Millions of possible visual variations
```

may require much more data.

---

# 🎯 12. Quantity vs Quality

This is a major lesson:

> **10 million bad examples aren't necessarily better than 100,000 excellent representative examples.**

Consider:

```text
Dataset A
1,000,000 images
↓
Blurry
Wrong labels
Duplicates
Biased
```

versus:

```text
Dataset B
100,000 images
↓
Correct labels
Diverse
Representative
Good quality
```

Dataset B may be far more useful.

---

# 🧹 13. Data Collection Should Match Deployment

Suppose you're building a road-sign detector.

Training data:

```text
☀️ Perfect daylight
Clean roads
High-quality camera
No obstruction
```

Deployment:

```text
🌧️ Rain
🌙 Night
🚗 Occlusion
📷 Cheap camera
Motion blur
```

Your model may struggle.

Therefore:

```text
        TRAINING WORLD
              ↓
        should resemble
              ↓
        DEPLOYMENT WORLD
```

This is one of the most important principles in practical ML.

---

# 📸 14. Your Computer Vision Dataset Example

Imagine your team collects traffic frames.

Raw source:

```text
Video
  ↓
Frame extraction
  ↓
Millions of frames
```

But you shouldn't immediately call those frames:

> "The dataset."

Instead:

```text
RAW DATA
   ↓
Remove corrupt frames
   ↓
Remove duplicates
   ↓
Select useful frames
   ↓
Annotate
   ↓
Quality checking
   ↓
Train/Val/Test split
   ↓
FINAL ML DATASET
```

This connects directly to the dataset/annotation workflow you were exploring earlier.

---

# 🏷️ 15. Annotation

For supervised ML, labels are extremely important.

For object detection:

```text
Image
 +
Bounding box
 +
Class
```

Example:

```text
┌─────────────────────────────┐
│                             │
│    ┌─────────────┐          │
│    │    CAR      │          │
│    └─────────────┘          │
│                             │
└─────────────────────────────┘
```

The annotation tells the model:

> "This region contains a car."

---

# ⚠️ 16. Bad Labels = Bad Learning

Suppose:

```text
Actual: CAR
Label: TRUCK
```

The model receives contradictory information.

Imagine thousands of such mistakes.

The model has difficulty learning the correct relationship.

Therefore:

$$
\boxed{\text{Label quality matters enormously}}
$$

---

# 👥 17. Human Annotation

Many ML datasets require humans.

For example:

```text
10,000 images
      ↓
Annotators
      ↓
Bounding boxes
      ↓
Quality control
```

For medical images, annotation may require trained professionals.

For physics experiments, labels may come from experimental conditions or simulations.

---

# 🔬 18. Real Data vs Synthetic Data

Sometimes collecting real data is difficult.

We can generate synthetic data.

For example:

```text
Physics simulation
      ↓
Millions of simulated measurements
      ↓
Training data
```

Computer vision:

```text
3D simulator
     ↓
Synthetic images
     ↓
Training
```

This can be useful, but synthetic data may differ from reality.

That's often called the:

> **Sim-to-real gap**

---

# 🔐 19. Data Privacy

Data collection isn't simply:

> "Collect everything."

You must consider:

```text
Privacy
Security
Consent
Legal requirements
Data ownership
Sensitive information
```

For example, collecting faces, medical records, or financial transactions has serious privacy implications.

---

# 🧠 20. The Complete Problem → Data Pipeline

Here's the diagram I want you to remember:

```text
                    REAL-WORLD PROBLEM
                           │
                           ↓
                  ┌──────────────────┐
                  │ Define Objective │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ What is X?       │
                  │ What is y?       │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ ML Problem Type  │
                  └────────┬─────────┘
                           ↓
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
         Regression   Classification  Clustering
             │             │
             └─────────────┼─────────────┘
                           ↓
                  ┌──────────────────┐
                  │ Data Collection  │
                  └────────┬─────────┘
                           ↓
               ┌───────────┴───────────┐
               ↓                       ↓
          Labeled Data            Unlabeled Data
               │                       │
               ↓                       ↓
          Supervised              Unsupervised
               │
               ↓
             DATASET
               │
               ↓
       Next: Data Understanding
```

---

# 🎯 21. A Complete Example

Let's practice the entire thought process.

## Problem

> "A hospital wants to predict whether a patient is at high risk of a particular condition."

### Step 1 — Objective

Predict:

```text
High Risk / Low Risk
```

### Step 2 — Features

Potentially:

```text
Age
Blood pressure
Lab measurements
Medical history
Other appropriate variables
```

### Step 3 — Target

```text
Risk = High / Low
```

### Step 4 — Problem Type

Output is a category:

$$
\boxed{\text{Classification}}
$$

### Step 5 — Data

Historical patient records with appropriate permissions and safeguards.

### Step 6 — Labels

We need reliable information indicating the outcome/risk status according to the defined prediction target.

### Step 7 — Prediction point

We must define **when** the prediction is made.

For example:

> At hospital admission.

Therefore, only information available at admission can be used.

This prevents leakage.

---

# 🔥 22. Another Example — Your ML Journey

Suppose you want:

> **Predict quantum decoherence/noise characteristics from experimental data.**

We can define:

```text
              QUANTUM EXPERIMENT
                      │
                      ↓
              Measurement Data
                      │
                      ↓
              Feature Extraction
                      │
                      ↓
              ┌───────────────┐
              │ ML FEATURES X │
              └───────┬───────┘
                      ↓
                    MODEL
                      ↓
              ┌───────────────┐
              │ TARGET y      │
              │ Noise /       │
              │ Decoherence   │
              └───────────────┘
```

Then we must determine:

* What exactly is the target?
* Is it numerical or categorical?
* What measurements are available before prediction?
* How much experimental data do we have?
* How noisy are the measurements?
* How representative are the experiments?
* Are simulations being used?
* Does simulated data match experimental data?

Those questions are **ML engineering questions**, even though the underlying problem is physics.

---

# 🧠 23. The Most Important Habit

Whenever you see a new ML problem, **don't immediately ask:**

> "Which algorithm should I use?"

Instead ask:

```text
1. What exactly is the problem?
             ↓
2. What am I predicting?
             ↓
3. What is X?
             ↓
4. What is y?
             ↓
5. When must the prediction be made?
             ↓
6. What data is available at that time?
             ↓
7. How will I collect it?
             ↓
8. Is the data representative?
             ↓
9. What metric defines success?
             ↓
10. Only THEN → choose a model
```

This mindset separates an **ML practitioner** from someone who simply knows how to call `fit()`.

---

# 🧠 One Final Reference Diagram

Put this in your notes:

```text
╔══════════════════════════════════════════════════════╗
║                 ML PROBLEM FORMULATION                ║
╚══════════════════════════════════════════════════════╝

                 REAL-WORLD QUESTION
                         │
                         ▼
              ┌──────────────────────┐
              │ What do we want to   │
              │ predict/decide?      │
              └──────────┬───────────┘
                         │
                         ▼
                  DEFINE TARGET y
                         │
                         ▼
              ┌──────────────────────┐
              │ What information is  │
              │ available?           │
              └──────────┬───────────┘
                         │
                         ▼
                 DEFINE FEATURES X
                         │
                         ▼
              ┌──────────────────────┐
              │ What type of output? │
              └──────────┬───────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        NUMBER       CATEGORY      GROUPS
            │            │            │
            ▼            ▼            ▼
       REGRESSION   CLASSIFICATION  CLUSTERING
                         │
                         ▼
                 COLLECT DATA
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        Database       API         Sensors
            │            │            │
            └────────────┼────────────┘
                         ▼
                    RAW DATA
                         │
                         ▼
                 NEXT STAGE
                         │
                         ▼
              DATA UNDERSTANDING
                    / EDA
```

---

## 🔜 Next Lesson: Data Understanding & EDA

Now we have **raw data**.

But raw data is almost never ready for ML.

Next we'll learn how an ML engineer looks at a dataset and asks:

> **"What the hell is inside this data?"** 😄

We'll cover **rows vs columns, numerical vs categorical vs ordinal vs binary data, distributions, missing values, duplicates, outliers, correlations, class imbalance, data visualization, and how Pandas/NumPy fit into EDA**, using a realistic dataset from start to finish.
