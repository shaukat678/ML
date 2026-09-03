# 🧹 ML Development Lifecycle — Part 4: Data Cleaning & Preparation

We've reached a **very important stage**.

So far:

```text
1. Problem Definition
        ↓
2. Data Collection
        ↓
3. EDA / Data Understanding
        ↓
4. DATA CLEANING & PREPARATION  ← TODAY
        ↓
5. Feature Engineering
        ↓
6. Model Selection
        ↓
...
```

Think of it like preparing food:

> **Raw data is the raw food. EDA tells you what's wrong with it. Data cleaning prepares it so the model can actually consume it.** 🍳

---

# 1. What Is Data Cleaning?

Data cleaning means finding and handling problems such as:

```text
Raw Data
   │
   ├── Missing values
   ├── Duplicate records
   ├── Invalid values
   ├── Wrong data types
   ├── Inconsistent categories
   ├── Outliers
   └── Formatting problems
          ↓
     Clean Data
```

But there's an important distinction:

### EDA

> **Discover the problem.**

### Data Cleaning

> **Decide what to do about the problem.**

For example:

```text
EDA:
"Salary has 8% missing values."

        ↓

Cleaning:
"Let's fill missing salaries using the median."
```

---

# 2. The Golden Principle

Don't blindly clean data.

You need to ask:

> **"What does this value mean in the real world?"**

Consider:

```text
Age = 150
```

Clearly suspicious.

But:

```text
Salary = 1,000,000
```

might be perfectly legitimate.

So cleaning is not simply:

```text
Find strange value → delete it
```

Instead:

```text
Find suspicious value
        ↓
Investigate
        ↓
Understand its meaning
        ↓
Choose appropriate treatment
```

---

# 3. Missing Values

Let's start with one of the most common problems.

```text
Age     Salary
25      50000
30      70000
NaN     60000
28      NaN
```

We have two missing values.

What can we do?

```text
Missing Value
      │
      ├───────────────┐
      ↓               ↓
   Remove          Fill
   it               it
                    │
              ┌─────┼─────┐
              ↓     ↓     ↓
            Mean  Median  Mode
```

---

# 4. Option 1 — Remove the Row

Suppose:

```text
Age     Salary
25      50000
30      70000
NaN     60000   ←
28      80000
```

We could remove that row.

Conceptually:

```text
Before:
4 rows

After:
3 rows
```

This can be reasonable when:

* only a tiny fraction is missing
* the missing rows aren't important
* enough data remains

But dangerous when:

```text
1,000,000 rows
500,000 have missing values
```

You can't simply throw away half your dataset.

---

# 5. Option 2 — Mean Imputation

Suppose:

```text
Age:

20
22
24
26
NaN
```

Mean:

$$
\frac{20+22+24+26}{4}=23
$$

So:

```text
NaN → 23
```

But there's a problem.

Consider:

```text
20
22
24
26
200
NaN
```

The mean is now heavily influenced by `200`.

That's why mean isn't always appropriate.

---

# 6. Option 3 — Median Imputation ⭐

For skewed data or data containing outliers, median is often more robust.

Example:

```text
Salaries:

40k
45k
50k
55k
2,000k
NaN
```

The median is around the normal range, while the mean gets pulled toward 2,000k.

So:

```text
Mean
  ↓
sensitive to extreme values

Median
  ↓
more resistant to extreme values
```

This is why you'll frequently see:

> **Median imputation for numerical variables with skew/outliers.**

But remember: it's a strategy, not a universal rule.

---

# 7. Option 4 — Mode Imputation

For categorical data:

```text
Car_Type

Sedan
SUV
Sedan
Sedan
NaN
SUV
```

The most frequent category is:

```text
Sedan
```

So:

```text
NaN → Sedan
```

That's called **mode imputation**.

---

# 8. But There's a BIG Problem 🚨

Imagine:

```text
Customer Income

50k
52k
48k
NaN
55k
```

If you calculate the median using the entire dataset **before splitting into train/test**, you've allowed information from the test set to influence your training process.

This is a form of **data leakage**.

Correct workflow:

```text
Raw Data
   ↓
Train / Test Split
   ↓
     ┌─────────────┐
     │ Training    │
     │ data        │
     └──────┬──────┘
            ↓
      Calculate median
            ↓
      Learn preprocessing
            ↓
     Apply same median
       ┌────┴─────┐
       ↓          ↓
     Train       Test
```

The preprocessing parameters should generally be **learned from the training data only**.

This idea will become extremely important when we study `scikit-learn` pipelines.

---

# 9. Duplicates

Suppose:

```text
ID   Age   Salary
1    25    50000
2    30    70000
3    25    50000
```

Rows 1 and 3 might be duplicates.

Possible workflow:

```text
Detect duplicates
       ↓
Investigate
       ↓
Are they genuinely duplicate?
       ↓
      Yes
       ↓
Remove
```

But again:

> Don't delete duplicates blindly.

Two people can legitimately have:

```text
Age = 25
Salary = 50000
```

Identical values don't necessarily mean identical records.

---

# 10. Invalid Values

Suppose we're building a house-price model.

```text
Bedrooms

2
3
4
-1
5
```

`-1` bedrooms doesn't make physical sense.

Another example:

```text
Age

20
25
31
-5
200
```

These are likely invalid.

So we may establish **domain constraints**:

```text
Age > 0
Bedrooms >= 0
Distance >= 0
Passengers >= 0
```

This is where your understanding of the **real-world problem** becomes extremely valuable.

---

# 11. Inconsistent Categories

This is a very common real-world problem.

Imagine:

```text
City

Karachi
karachi
KARACHI
Karachi 
Karachi, Pakistan
```

Humans understand these might mean the same thing.

A computer sees different strings.

```text
"Karachi"
"karachi"
"KARACHI"
"Karachi "
```

We need to standardize them.

Conceptually:

```text
Raw categories
      ↓
Normalize formatting
      ↓
Standard categories
```

For example:

```text
"KARACHI" → "karachi"
"Karachi" → "karachi"
"Karachi " → "karachi"
```

---

# 12. Units Can Also Be Inconsistent

Suppose:

```text
Distance

5 km
10 miles
3 km
```

If you treat them all as simply `5`, `10`, `3`, you're making a serious mistake.

You need a common unit:

```text
km
 ↓
Convert everything
 ↓
km
```

For example:

$$
1\text{ mile}\approx1.609\text{ km}
$$

Data cleaning therefore isn't only about missing values.

It includes **semantic consistency**.

---

# 13. Wrong Data Types

Imagine:

```text
Age

"25"
"30"
"40"
"35"
```

They look like numbers.

But the computer may interpret them as:

```text
string/object
```

rather than:

```text
integer
```

This matters because:

```text
"100" + "20"
```

can mean string concatenation:

```text
"10020"
```

rather than:

```text
120
```

So we sometimes need type conversion:

```text
String
  ↓
Numeric
```

---

# 14. Dates Are Special

Imagine:

```text
2026-01-05
2026-02-10
2026-03-15
```

Dates contain a lot of information.

You may extract:

```text
Year
Month
Day
Day of week
Hour
Weekend/weekday
```

For example:

```text
2026-09-03 07:30

        ↓

Year  = 2026
Month = 9
Day   = 3
Hour  = 7
```

This moves us toward **feature engineering**, which we'll study separately.

---

# 15. Outliers — Clean or Keep?

Remember our earlier example:

```text
Salary:

40k
45k
50k
55k
60k
900k
```

Should we delete 900k?

**Not automatically.**

Maybe it's:

```text
CEO
        ↓
900k is legitimate
```

Or:

```text
Typing mistake
        ↓
900k should have been 90k
```

Therefore:

```text
Outlier
   ↓
Investigate
   ↓
┌───────────────┐
│ Real value?   │
└───────┬───────┘
        │
   ┌────┴────┐
   ↓         ↓
  Yes        No
   ↓         ↓
 Keep      Correct/
           Remove
```

---

# 16. Three Different Things You Must Not Confuse

This is a very important distinction:

### Outlier

A value unusually far from others.

```text
10 11 12 13 100
```

### Invalid value

A value that violates domain rules.

```text
Age = -10
```

### Rare value

A value that occurs infrequently but may be completely legitimate.

```text
Disease = rare disease
```

These are **not the same thing**.

---

# 17. Scaling

Now we reach something you've already encountered:

> **Normalization and Standardization**

Suppose:

```text
Age        = 25
Salary     = 150000
Distance   = 12
```

These features operate on very different numerical scales.

```text
Age       → tens
Distance  → tens
Salary    → hundreds of thousands
```

Some algorithms are sensitive to feature scale.

We can transform them.

---

# 18. Standardization

Standardization generally transforms a feature using:

$$
z=\frac{x-\mu}{\sigma}
$$

where:

* \(x\) = original value
* \(\mu\) = mean
* \(\sigma\) = standard deviation

The resulting feature typically has:

```text
mean ≈ 0
std  ≈ 1
```

Conceptually:

```text
Raw Feature
     ↓
subtract mean
     ↓
divide by std
     ↓
Standardized Feature
```

---

# 19. Normalization

One common form is Min-Max scaling:

$$
x'=\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
$$

This maps values approximately into:

```text
0 → 1
```

Example:

```text
Original:

10
20
30
40
50

       ↓

0
0.25
0.50
0.75
1
```

You've already studied this topic, so later we'll connect it directly to **when scaling is necessary and when it isn't**.

---

# 20. Very Important: Not Every Model Needs Scaling

This is something beginners often misunderstand.

For example, many tree-based models:

```text
Decision Tree
Random Forest
Gradient Boosting
```

are generally not dependent on feature magnitude in the same way distance-based or gradient-based models are.

Whereas scaling can be important for models such as:

```text
KNN
K-Means
SVM
Neural Networks
Logistic Regression
Linear Regression (especially with regularization)
```

The exact preprocessing requirement depends on the algorithm and training setup.

---

# 21. Encoding Categorical Data

Suppose:

```text
Car_Type

Sedan
SUV
Truck
```

Most ML algorithms need numerical representations.

So:

```text
Categorical
    ↓
Encoding
    ↓
Numerical representation
```

One common method is **One-Hot Encoding**.

```text
Car_Type

Sedan
SUV
Truck
```

becomes approximately:

```text
Sedan   SUV   Truck
  1      0      0
  0      1      0
  0      0      1
```

This prevents the model from automatically assuming:

```text
Truck > SUV > Sedan
```

when no such order exists.

---

# 22. Label Encoding

Another possibility:

```text
Sedan → 0
SUV   → 1
Truck → 2
```

This can be appropriate in certain situations, particularly when the categories are genuinely ordinal or when the algorithm/encoding setup is designed for categorical values.

But blindly assigning numbers to nominal categories can create a fake relationship.

For example:

```text
Red   → 0
Blue  → 1
Green → 2
```

Does that mean:

$$
Green > Blue > Red
$$

❌ No.

That's why encoding requires thought.

---

# 23. The Complete Cleaning Pipeline

Now put everything together:

```text
                    RAW DATA
                       │
                       ▼
             ┌──────────────────┐
             │ Check data types  │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Missing values   │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Duplicate rows   │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Invalid values   │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Inconsistencies  │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Outlier handling │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Encoding         │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Scaling          │
             └────────┬─────────┘
                      ↓
                 CLEAN DATA
```

But there is one subtle correction:

**Encoding and scaling are often considered preprocessing rather than "cleaning" strictly speaking.** In practical ML workflows, however, they're commonly discussed together under **data preparation/preprocessing**.

---

# 24. The Most Important Concept: Train/Test Leakage

Let's make this crystal clear.

Suppose you have:

```text
1000 observations
```

You split:

```text
80% → Training
20% → Testing
```

Correct:

```text
                 Dataset
                    │
             Train/Test Split
              /           \
             ↓             ↓
         Training         Testing
             │
             ↓
      Learn preprocessing
             │
             ↓
       Apply transformation
          /         \
         ↓           ↓
      Training     Testing
```

The test set should remain a genuine simulation of **unseen future data**.

If you calculate statistics from the entire dataset first:

```text
Entire Dataset
     ↓
calculate mean
     ↓
split
```

then information from the test set has influenced your preprocessing.

That's leakage.

---

# 25. The Scikit-Learn Way

Eventually, you'll use pipelines like:

```python
from sklearn.pipeline import Pipeline
```

Conceptually:

```text
Raw Data
   ↓
Imputer
   ↓
Scaler
   ↓
Model
```

The pipeline learns the preprocessing from training data and applies the learned transformations consistently.

This becomes especially powerful in production.

We'll build these properly later.

---

# 26. A Real Example

Suppose our taxi dataset is:

| Distance | Traffic | Passengers | Car_Type | Fare |
| -------: | ------: | ---------: | -------- | ---: |
|        5 |       2 |          2 | Sedan    |  800 |
|        8 |       4 |          3 | SUV      | 1300 |
|      NaN |       2 |          1 | Sedan    |  600 |
|       10 |       3 |         -1 | SUV      | 1500 |
|        5 |       2 |          2 | Sedan    |  800 |

We discover:

### Problem 1

```text
Distance = NaN
```

Missing value.

### Problem 2

```text
Passengers = -1
```

Invalid.

### Problem 3

Rows 1 and 5 may be duplicates.

### Problem 4

`Car_Type` is categorical.

### Problem 5

Numerical features have different scales.

So:

```text
Raw Dataset
     │
     ├── Missing distance
     │       ↓
     │    Imputation
     │
     ├── Invalid passengers
     │       ↓
     │    Investigate/fix
     │
     ├── Duplicate row
     │       ↓
     │    Investigate/remove
     │
     ├── Car type
     │       ↓
     │    Encode
     │
     └── Different scales
             ↓
          Scale if needed
```

Now we have data that is much more suitable for modeling.

---

# 🧠 The Mental Model You Should Remember

Think of ML data preparation as **taking a messy laboratory sample and preparing it for an experiment**.

```text
RAW DATA
   │
   │  "What's wrong?"
   ↓
EDA
   │
   │  "Fix the problems"
   ↓
CLEANING
   │
   │  "Make it model-compatible"
   ↓
PREPROCESSING
   │
   ├── Imputation
   ├── Encoding
   ├── Scaling
   └── Transformations
   │
   ↓
MODEL-READY DATA
```

And remember these three words:

> **EDA discovers. Cleaning fixes. Preprocessing transforms.**

---

# 🔥 One More Important Idea

Data preparation isn't:

```text
"Make the data look beautiful."
```

It's:

> **Make the data accurately represent the real-world problem in a form the ML algorithm can learn from.**

That's the real goal.

---

## Where We Go Next

We've now reached the next major stage:

```text
Problem
   ↓
Data Collection
   ↓
EDA
   ↓
Cleaning & Preparation
   ↓
⭐ FEATURE ENGINEERING
   ↓
Model Selection
   ↓
Training
```

And **Feature Engineering** is especially important because this is where we answer:

> **“What information should we actually give the model?”**

We'll cover things like:

```text
Raw features
    ↓
Create new features
    ↓
Transform features
    ↓
Ratios
    ↓
Log transformations
    ↓
Date/time features
    ↓
Interaction features
    ↓
Domain-specific features
    ↓
Feature selection
```

This connects directly to the ideas you've previously encountered like **distance/traffic ratios, passenger-related features, peak-hour features, and log(salary)**.
