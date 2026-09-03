# 📊 ML Development Lifecycle — Part 3: Data Understanding & EDA

Now that we know **what problem we're solving and how to collect the data**, the next question is:

> **“What is actually inside our data?”**

This is where **EDA — Exploratory Data Analysis** comes in.

---

# 1. Where We Are in the ML Lifecycle

Remember our pipeline:

```text
┌─────────────────────┐
│ 1. Problem          │
│    Definition       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 2. Data Collection  │
└──────────┬──────────┘
           ↓
┌──────────────────────────────┐
│ 3. DATA UNDERSTANDING / EDA  │  ← TODAY
└──────────┬───────────────────┘
           ↓
┌─────────────────────┐
│ 4. Data Preparation │
└──────────┬──────────┘
           ↓
          ...
```

EDA is basically:

> **“Let me investigate my data before I trust it.”**

---

# 2. What is EDA?

**Exploratory Data Analysis** means examining and investigating a dataset to understand:

* What columns do we have?
* What does each column mean?
* What type of data is present?
* Are values missing?
* Are there duplicates?
* Are there strange values?
* Are there outliers?
* What are the distributions?
* Which features are related?
* Is the target balanced?
* Are there suspicious patterns?
* Is there potential data leakage?

The fundamental idea:

```text
Raw Dataset
     │
     ▼
┌─────────────────┐
│      EDA        │
│                 │
│ Understand      │
│ Investigate     │
│ Question        │
│ Visualize       │
│ Discover        │
└────────┬────────┘
         ▼
Better decisions
about the ML model
```

---

# 3. Think Like a Detective 🕵️

Imagine someone gives you this dataset:

| Distance | Traffic | Car_Type | Passengers | Fare |
| -------: | ------: | -------- | ---------: | ---: |
|      5.2 |       2 | Sedan    |          2 |  850 |
|      8.1 |       4 | SUV      |          3 | 1300 |
|      2.4 |       1 | Sedan    |          1 |  500 |
|     50.0 |       1 | SUV      |          2 |  700 |
|     -3.2 |       2 | Sedan    |          1 |  400 |

Would you immediately train a model?

**NO.**

You should investigate.

Look at row 4:

```text
Distance = 50 km
Traffic = 1
Fare = 700
```

Maybe that's legitimate.

But row 5:

```text
Distance = -3.2 km
```

🚨 Impossible.

So EDA helps us discover problems **before they damage our model**.

---

# 4. First Understand the Structure

Suppose our dataset looks like:

```text
        Dataset
           │
           ├─────────────── Rows
           │
           └─────────────── Columns
                            │
                            ├── Numerical
                            ├── Categorical
                            ├── Ordinal
                            ├── Binary
                            └── Date/Time
```

### Rows

Usually represent **observations/examples**.

For example:

```text
Row 1 → Taxi trip
Row 2 → Taxi trip
Row 3 → Taxi trip
...
```

### Columns

Usually represent **variables/features/target**.

```text
Distance
Traffic
Passengers
Car_Type
Fare
```

So:

```text
              Features             Target
                 ↓                    ↓
┌──────────┬─────────┬──────────┬─────────┐
│ Distance │ Traffic │ Car_Type │   Fare  │
├──────────┼─────────┼──────────┼─────────┤
│ 5.2      │ 2       │ Sedan    │ 850     │
│ 8.1      │ 4       │ SUV      │ 1300    │
│ 2.4      │ 1       │ Sedan    │ 500     │
└──────────┴─────────┴──────────┴─────────┘
       X                              y
```

---

# 5. Data Types — Very Important

Before doing anything with data, identify the types.

## Numerical Data 🔢

Numbers representing quantities.

Examples:

```text
Age
Salary
Distance
Temperature
Height
Weight
Number of passengers
```

Example:

```text
Age = 25
Distance = 12.5 km
Salary = 150000
```

Numerical data can be:

### Discrete

Countable:

```text
Passengers = 1, 2, 3, 4
Children = 0, 1, 2
```

### Continuous

Can take decimal values:

```text
Height = 172.5 cm
Temperature = 27.43°C
Distance = 12.73 km
```

---

# 6. Categorical Data 🏷️

Categories rather than quantities.

Example:

```text
Car_Type

Sedan
SUV
Hatchback
Truck
```

Even though we might represent them as:

```text
Sedan     → 0
SUV       → 1
Truck     → 2
```

those numbers **don't necessarily have mathematical meaning**.

This is important because later we'll discuss:

> One-hot encoding
> Label encoding
> Target encoding

---

# 7. Ordinal Data

This one is interesting.

Categories have a meaningful order.

Example:

```text
Customer Satisfaction

Bad
 ↓
Okay
 ↓
Good
 ↓
Excellent
```

There is an order:

```text
Bad < Okay < Good < Excellent
```

Unlike:

```text
Red
Blue
Green
```

where there is no natural order.

---

# 8. Binary Data

Only two possibilities:

```text
Yes / No

0 / 1

True / False

Spam / Not Spam
```

Example:

```text
Has_Credit_Card

1
0
1
1
0
```

---

# 9. First EDA Question: How Big Is My Dataset?

Suppose:

```text
1,000,000 rows
20 columns
```

That's very different from:

```text
500 rows
20 columns
```

Why?

Because the amount of data affects:

* model complexity
* training
* overfitting
* computational requirements
* statistical reliability

A basic Pandas investigation:

```python
df.shape
```

might return:

```text
(1000000, 20)
```

Meaning:

```text
1,000,000 rows
20 columns
```

---

# 10. What Are My Columns?

Use:

```python
df.columns
```

Example:

```text
Index([
    'distance',
    'traffic',
    'passengers',
    'car_type',
    'fare'
])
```

Then:

```python
df.head()
```

to see the first few rows.

And:

```python
df.info()
```

to get an overview of:

* column names
* data types
* number of non-null values

---

# 11. Missing Values 🚨

This is one of the most important EDA concepts.

Imagine:

| Age | Salary | Experience |
| --: | -----: | ---------: |
|  25 |  50000 |          2 |
|  31 |  70000 |          5 |
| NaN |  60000 |          4 |
|  28 |    NaN |          3 |

`NaN` means:

> **We don't have a value here.**

So:

```text
Dataset
   │
   ├── Complete values
   │
   └── Missing values
```

Why do missing values matter?

Because many ML algorithms cannot directly work with missing values.

---

# 12. Why Are Values Missing?

This is more important than simply counting them.

Suppose:

```text
Age = missing
```

Why?

Possibilities:

```text
Person didn't provide it
        ↓
Data collection problem
        ↓
Sensor failure
        ↓
Database error
        ↓
Value intentionally hidden
```

The reason matters.

For example, suppose a hospital dataset has:

```text
Income = missing
```

Maybe low-income patients are disproportionately less likely to report income.

Then missingness itself might contain information.

So don't blindly do:

```text
NaN → 0
```

❌ That's often wrong.

---

# 13. How Much Data Is Missing?

Suppose:

```text
Age        → 2% missing
Salary     → 4% missing
Experience → 60% missing
```

That immediately tells us:

```text
Age        🟢 probably manageable
Salary     🟢 probably manageable
Experience 🔴 serious issue
```

Later we'll learn different strategies:

```text
Missing Data
     │
     ├── Remove rows
     ├── Remove column
     ├── Mean/Median imputation
     ├── Mode
     ├── Forward/backward fill
     └── Model-based imputation
```

But **first EDA**, then decide.

---

# 14. Duplicate Rows

Imagine:

```text
Row 1 → Ali, 25, 50000
Row 2 → Sara, 30, 70000
Row 3 → Ali, 25, 50000
Row 4 → Ahmed, 28, 60000
```

Rows 1 and 3 might be duplicates.

If we accidentally count the same observation many times:

```text
Real data
   ↓
Duplicate rows
   ↓
Model sees repeated information
   ↓
Dataset becomes biased
```

We therefore investigate duplicates during EDA.

---

# 15. Outliers 🚨

This is extremely important.

Suppose salaries are:

```text
40k
45k
50k
55k
60k
58k
52k
900k
```

`900k` is dramatically different.

It might be an:

### Actual outlier

A CEO really earns 900k.

OR

### Data error

Someone accidentally entered:

```text
900000
```

instead of:

```text
90000
```

These are very different situations.

So:

> **An outlier is not automatically an error.**

---

# 16. Outlier Mental Model

Think:

```text
             Strange value
                  │
          ┌───────┴───────┐
          ↓               ↓
      Real event       Data error
          │               │
       Keep it         Investigate/fix
```

This distinction is critical in real ML projects.

---

# 17. Distribution 📈

Now we ask:

> **How are the values distributed?**

Suppose we have:

```text
Age:

18 20 21 22 23 24 25 25 26 27 28 ...
```

We might discover:

```text
Most people → 20–35
Few people  → 60+
```

That's the distribution.

---

# 18. Why Distribution Matters

Consider two datasets.

### Dataset A

```text
45 47 48 49 50 51 52 53 54 55
```

Very concentrated.

### Dataset B

```text
1 5 10 20 50 100 500 1000
```

Very spread out.

A model may behave very differently on these datasets.

Distribution helps us understand:

* central tendency
* spread
* skewness
* unusual values
* concentration
* possible transformations

---

# 19. Mean vs Median

Suppose:

```text
Salaries:

50k
55k
60k
65k
70k
2,000k
```

Mean becomes heavily influenced by:

```text
2,000k
```

while the median is much more stable.

This is why EDA helps us decide whether:

```text
Mean
```

or

```text
Median
```

better represents the data.

We'll later connect this directly to **imputation and preprocessing**.

---

# 20. Correlation 🔗

Now we start asking:

> **Are features related to each other?**

Suppose:

```text
Distance ↑
       ↓
Fare ↑
```

Maybe:

```text
distance ↑ → fare ↑
```

We can investigate the relationship.

Example:

```text
Distance     Fare

2 km         400
5 km         700
10 km        1200
20 km        2200
```

There appears to be a relationship.

---

# 21. Correlation ≠ Causation ⚠️

Very important.

Suppose we discover:

```text
Ice cream sales ↑
Drowning incidents ↑
```

Does ice cream cause drowning?

❌ No.

A third factor might be responsible:

```text
        HOT WEATHER
        /         \
       ↓           ↓
Ice cream      Swimming
sales ↑        ↑
              ↓
        Drowning ↑
```

So correlation means:

> **Two variables tend to change together.**

It does NOT automatically mean:

> **One causes the other.**

---

# 22. Feature–Target Relationship

This is particularly important for ML.

Suppose:

```text
X:

Distance
Traffic
Passengers
Car_Type

y:

Fare
```

We want to understand:

```text
Distance ────────┐
Traffic ─────────┤
Passengers ──────┼──→ Fare
Car_Type ────────┘
```

Questions:

```text
Does distance affect fare?

Does traffic affect fare?

Does car type affect fare?

Does passenger count matter?
```

EDA helps us investigate these questions.

---

# 23. Class Imbalance

This matters in **classification**.

Suppose we're detecting fraud.

Dataset:

```text
Normal transactions → 99,000
Fraud transactions   → 1,000
```

That's:

```text
Normal = 99%
Fraud  = 1%
```

This is **class imbalance**.

Imagine a stupid model:

```text
Predict "Normal" EVERY TIME
```

Accuracy:

```text
99%
```

😱 Sounds excellent!

But it detects:

```text
0% of fraud
```

So:

> **Accuracy alone can be extremely misleading on imbalanced datasets.**

We'll later study:

* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC
* PR-AUC

---

# 24. Data Leakage During EDA 🚨

Here's a subtle but extremely important concept.

Suppose we're predicting:

```text
Will customer cancel subscription?
```

Features:

```text
Age
Plan
Usage
Complaints
```

But someone accidentally includes:

```text
Cancellation_Date
```

That's information that only exists **after cancellation**.

Then:

```text
Past/current information ──→ Prediction
                              ↑
                        allowed

Future information ───────────→
                        NOT allowed
```

This is **data leakage**.

The model appears amazing during testing but fails in the real world.

---

# 25. EDA Is Not Just Looking at Numbers

EDA has two major sides:

```text
              EDA
             /   \
            /     \
       Numerical   Visual
       Analysis    Analysis
          │           │
          ↓           ↓
      statistics   graphs
```

### Numerical

```python
df.describe()
```

can show:

```text
count
mean
std
min
25%
50%
75%
max
```

### Visual

You might use:

```text
Histogram
Box plot
Scatter plot
Bar chart
Correlation heatmap
```

---

# 26. The Most Important EDA Visualizations

Think of them as your detective tools 🔍.

### Histogram

Answers:

> How is one numerical variable distributed?

```text
Frequency
  │
  │      ███
  │    ███████
  │  ███████████
  │██████████████
  └──────────────────→ Age
```

---

### Box Plot

Useful for:

* spread
* median
* quartiles
* outliers

```text
       ───────
         │
    ┌───────────┐
────│    │      │────
    └───────────┘
         │
       ─────
          •
          •  ← possible outliers
```

---

### Scatter Plot

Answers:

> How do two numerical variables relate?

```text
Fare
 │
 │             •
 │          •
 │       •
 │    •
 │ •
 └────────────────── Distance
```

---

### Bar Chart

Great for categorical data:

```text
Number of customers

Sedan      █████████████
SUV        ████████
Truck      ████
Hatchback  ██████
```

---

# 27. A Real EDA Workflow

When you receive a new dataset, follow this:

```text
                 DATASET
                    │
                    ▼
             ┌─────────────┐
             │ Understand  │
             │ the schema  │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Check size  │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Data types  │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Missing     │
             │ values      │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Duplicates  │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Statistics  │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Distributions│
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Outliers    │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Relationships│
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Target      │
             │ analysis    │
             └──────┬──────┘
                    ↓
             ┌─────────────┐
             │ Leakage /   │
             │ quality     │
             └──────┬──────┘
                    ↓
             READY FOR
          DATA PREPARATION
```

---

# 28. The Most Important Mindset

Don't think:

> "EDA means making some graphs."

That's too shallow.

Think:

> **EDA = interrogating your dataset.**

Ask your data:

```text
Who are you?

What do your columns mean?

How much data do I have?

Are values missing?

Are there errors?

Are there duplicates?

Are there outliers?

How are values distributed?

Which variables are related?

Is my target balanced?

Is there leakage?

Does this data actually represent the real world?
```

That's EDA.

---

# 🧠 The Big Picture

You should now mentally see:

```text
REAL WORLD
    │
    ▼
Problem
    │
    ▼
Collect Data
    │
    ▼
┌─────────────────────────────┐
│          EDA                │
│                             │
│  What do I actually have?   │
│                             │
│  ├─ Shape                   │
│  ├─ Columns                 │
│  ├─ Data Types              │
│  ├─ Missing Values          │
│  ├─ Duplicates              │
│  ├─ Outliers                │
│  ├─ Distributions           │
│  ├─ Correlations            │
│  ├─ Target                  │
│  ├─ Imbalance               │
│  └─ Leakage                 │
└──────────────┬──────────────┘
               ↓
       Understand the data
               ↓
       Data Preparation
               ↓
        Feature Engineering
               ↓
          Model Training
```

### ⭐ Golden Rule

> **Never blindly feed raw data into an ML algorithm.**

First:

**Understand it → question it → visualize it → validate it.**

Then prepare it.

---

## Next lesson

We'll move to **Data Cleaning & Preparation**, where we'll take everything discovered during EDA and actually fix it:

```text
Missing Values
      ↓
Duplicates
      ↓
Invalid Values
      ↓
Outliers
      ↓
Data Types
      ↓
Categorical Data
      ↓
Scaling
      ↓
Clean Dataset
```

And this is where **Pandas + NumPy** that you've already been learning start becoming extremely powerful in an actual ML workflow.
