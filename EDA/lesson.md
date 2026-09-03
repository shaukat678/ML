
---

# Part 3 — Exploratory Data Analysis (EDA) 🔎

## What you should be able to do after this

By the end, you should be able to take a dataset you've never seen before and systematically answer:

```text
What data do I have?
        ↓
What does each column mean?
        ↓
Is the data trustworthy?
        ↓
What is missing?
        ↓
What is unusual?
        ↓
How are variables distributed?
        ↓
How do features relate to the target?
        ↓
Are there hidden patterns?
        ↓
Are classes imbalanced?
        ↓
Is there leakage?
        ↓
Are train/test distributions different?
        ↓
What should I do next?
```

That is **real EDA**.

---

# 1. First: What Exactly Is EDA?

EDA = **Exploratory Data Analysis**.

Break the words apart:

### Exploratory

You're investigating something you don't completely understand.

### Data

Your dataset.

### Analysis

You're trying to discover patterns, problems, relationships and useful information.

So:

> **EDA is the systematic investigation of a dataset before building a reliable ML model.**

Think of yourself as a **data detective** 🕵️.

You receive:

```text
customer_data.csv
```

You don't immediately do:

```python
model.fit(X, y)
```

Instead:

```text
                 DATASET
                    ↓
              🔎 INVESTIGATE
                    ↓
       ┌────────────┼────────────┐
       ↓            ↓            ↓
    Structure    Quality       Patterns
       ↓            ↓            ↓
    columns      missing       distributions
    dtypes       duplicates     relationships
    shape        invalid        outliers
                                 ↓
                            Target behavior
                                 ↓
                              Leakage?
```

Only after this do you start serious modeling.

---

# 2. A Real-World Scenario

Imagine you're hired by a food-delivery company.

Your task:

> **Predict whether a customer will cancel their order.**

You receive:

```text
orders.csv
```

with:

| order_id | distance_km | delivery_time | restaurant_rating | customer_age | weather | cancelled |
| -------- | ----------: | ------------: | ----------------: | -----------: | ------- | --------- |
| 101      |         2.4 |            28 |               4.5 |           22 | Clear   | 0         |
| 102      |         8.1 |            55 |               3.2 |           31 | Rain    | 1         |
| 103      |         1.2 |            20 |               4.8 |           25 | Clear   | 0         |
| ...      |         ... |           ... |               ... |          ... | ...     | ...       |

You might think:

> "Easy. Train Random Forest."

No.

Your first job is to **interrogate the dataset**.

---

# 3. The EDA Workflow

Memorize this:

```text
                    EDA
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
  1 Structure     2 Quality      3 Statistics
       │             │             │
       ↓             ↓             ↓
  shape/dtypes    missing        mean/median
  columns         duplicates     std/quantiles
  target          invalid        distribution
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                4 Visualize
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
     Univariate   Bivariate   Multivariate
          │          │          │
          └──────────┼──────────┘
                     ↓
                5 Relationships
                     ↓
                6 Outliers
                     ↓
                7 Target
                     ↓
                8 Leakage
                     ↓
                9 Train/Test Drift
                     ↓
              EDA conclusions
```

---

# 4. Step 1 — Understand the Dataset Structure

The first question:

> **What exactly did I receive?**

Start with:

```python
df.shape
```

Suppose:

```text
(100000, 15)
```

This means:

```text
100,000 rows
15 columns
```

Think:

> **Rows = observations**

> **Columns = variables/features**

For example:

```text
100,000 customers
15 pieces of information about each customer
```

---

# 5. Inspect the First Rows

```python
df.head()
```

This lets you see what the data actually looks like.

Don't underestimate this.

You might discover:

```text
age
25
31
"unknown"
42
```

or:

```text
salary
50000
60000
"70k"
```

or:

```text
city
Karachi
karachi
KARACHI
```

Already we have potential problems.

---

# 6. Inspect the Last Rows

```python
df.tail()
```

Why?

Sometimes data problems occur at the end because of:

* incomplete imports
* appended records
* corrupted rows
* different data sources

---

# 7. Column Names

```python
df.columns
```

You want to understand:

```text
What does every column represent?
```

For example:

```text
[
 'age',
 'salary',
 'city',
 'experience',
 'education',
 'purchased'
]
```

But don't just trust the names.

A column called:

```text
income
```

might actually mean:

> monthly income

rather than:

> annual income

Domain understanding matters.

---

# 8. Data Types

Use:

```python
df.info()
```

You might see:

```text
age             int64
salary          float64
city            object
experience      int64
purchased       int64
```

Now classify them conceptually.

### Numerical

```text
age
salary
height
temperature
distance
```

### Categorical

```text
city
gender
browser
product_type
```

### Ordinal

Categories with meaningful order:

```text
education:
High School < Bachelor < Master < PhD
```

### Binary

```text
purchased:
0 / 1
```

### Date/time

```text
2026-08-17 14:32:10
```

This classification matters because **different types require different EDA and preprocessing strategies**.

---

# 9. A Common Beginner Mistake

Suppose:

```python
df["age"].dtype
```

returns:

```text
object
```

You might think:

> "Age is categorical."

Not necessarily.

Maybe your column contains:

```text
"21"
"25"
"30"
"unknown"
"42"
```

The **semantic type** is numerical, even though the stored data type is object/string.

That's an important distinction.

---

# 10. Basic Statistical Summary

Now:

```python
df.describe()
```

You might get:

|       |   age |    salary | distance |
| ----- | ----: | --------: | -------: |
| count | 98000 |     97000 |   100000 |
| mean  |  34.2 |     85000 |     12.4 |
| std   |   9.5 |     42000 |     18.1 |
| min   |    18 |     18000 |      0.2 |
| 25%   |    27 |     55000 |      4.2 |
| 50%   |    32 |     72000 |      8.1 |
| 75%   |    40 |     98000 |     15.4 |
| max   |    92 | 2,000,000 |      500 |

This table contains a **lot** of information.

---

# 11. Understand Each Statistic

## Count

How many non-missing observations?

```text
salary count = 97,000
```

But dataset has:

```text
100,000 rows
```

Therefore:

```text
3,000 salary values are missing
```

Already discovered something.

---

# 12. Mean

$$
Mean = \frac{\sum x_i}{n}
$$

Example:

```text
10, 20, 30
```

Mean:

$$
20
$$

But mean can be strongly affected by extreme values.

---

# 13. Median

Sort:

```text
10, 20, 30, 40, 1000
```

Median:

```text
30
```

Mean:

```text
220
```

Huge difference.

Why?

Because:

```text
1000
```

is pulling the mean upward.

Therefore:

> **Median is often more robust to extreme values than mean.**

---

# 14. Standard Deviation

Standard deviation tells you roughly:

> **How spread out the values are around the mean.**

Low:

```text
48
50
51
49
50
```

High:

```text
10
50
90
20
100
```

You don't just memorize the formula.

Think:

> **Mean tells me where the data is centered.**

> **Standard deviation tells me how spread out it is.**

---

# 15. Percentiles — Extremely Important

`describe()` gives:

```text
25%
50%
75%
```

These are percentiles.

### 50th percentile

Median.

### 25th percentile

25% of observations are below this value.

### 75th percentile

75% are below this value.

Example:

```text
salary:

25% → 50,000
50% → 70,000
75% → 100,000
```

This tells us about the distribution.

---

# 16. IQR

Interquartile Range:

$$
IQR = Q_3 - Q_1
$$

For:

```text
Q1 = 50,000
Q3 = 100,000
```

we get:

$$
IQR = 50,000
$$

IQR is heavily used when investigating outliers.

---

# 17. Missing Values — Your First Major Investigation

Use:

```python
df.isnull().sum()
```

Example:

```text
age                 120
salary             3000
city                 45
experience            0
restaurant_rating    850
```

Now don't immediately say:

> "Fill everything with mean."

🚨 That's poor EDA.

First ask:

> **Why is the data missing?**

---

# 18. Missingness Has Meaning

Suppose:

```text
income
```

is missing for many customers.

Possibility:

> High-income customers intentionally didn't disclose income.

Then missingness itself may contain information.

Another example:

```text
medical_test_result = NaN
```

Maybe the test wasn't required.

Different reason → different treatment.

---

# 19. Missing Data Mechanisms

At intermediate level, learn these concepts:

### MCAR

**Missing Completely At Random**

Missingness unrelated to observed or unobserved values.

### MAR

**Missing At Random**

Missingness depends on other observed variables.

### MNAR

**Missing Not At Random**

Missingness depends on the missing value itself or an unobserved factor.

You don't always need to mathematically prove which mechanism applies, but understanding the idea is valuable.

---

# 20. Duplicate Rows

Check:

```python
df.duplicated().sum()
```

Suppose:

```text
2500
```

duplicates.

Don't blindly delete them.

Ask:

> Are these truly duplicate observations?

For example, two identical bank transactions could actually be legitimate.

But if the same customer record was accidentally imported twice, that's a problem.

---

# 21. Invalid Values

This is one of the most important EDA skills.

Imagine:

```text
age
22
31
45
-5
200
```

Technically:

```text
-5
```

is a number.

Statistically valid.

But **domain-invalid**.

That's why EDA isn't just statistics.

You need domain knowledge.

---

# 22. Domain Constraints

Examples:

```text
age >= 0

height > 0

temperature may be negative depending on domain

distance >= 0

number_of_children >= 0

price > 0

probability ∈ [0,1]
```

For a physics student, this should feel natural.

You already think in terms of:

$$
m>0
$$

$$
t\geq0
$$

$$
0\leq P\leq1
$$

EDA is partly **physical/domain sanity checking**.

---

# 23. Categorical Inconsistencies

Suppose:

```python
df["city"].value_counts()
```

gives:

```text
Karachi       40,000
Lahore        30,000
Islamabad     20,000
karachi          500
KARACHI          300
Karachi           70
```

That's potentially one category represented in multiple ways.

You may need normalization:

```text
Karachi
karachi
KARACHI
Karachi 
```

→

```text
Karachi
```

But again, understand the domain before changing values.

---

# 24. Univariate Analysis

Now we move to the heart of EDA.

**Univariate** means:

> Analyze one variable at a time.

Examples:

```text
age alone
salary alone
city alone
```

Questions:

```text
What is its distribution?
Is it skewed?
Are there outliers?
Is it concentrated?
Are there rare categories?
```

---

# 25. Histogram 📊

For numerical data:

```python
df["salary"].hist()
```

A histogram shows the distribution.

Imagine:

```text
frequency
  │
  │       ███
  │     ███████
  │   ██████████
  │ █████████████
  └──────────────── salary
```

You can visually see where observations are concentrated.

---

# 26. Why Distribution Matters

Consider:

```text
salary
```

If:

```text
most people → 30k–100k
few people  → 1M+
```

the distribution is **right-skewed**.

This may influence:

* imputation
* transformations
* visualization
* model choice
* outlier treatment

---

# 27. Skewness

### Symmetric

Approximately:

```text
      █
    ████
   ██████
    ████
      █
```

### Right-skewed

```text
     ███
   █████
 ███████
████████
         ███
            ██
              █
```

Long tail toward larger values.

### Left-skewed

Long tail toward smaller values.

---

# 28. Log Transformation

Suppose salaries range:

```text
30,000
50,000
80,000
100,000
2,000,000
```

Large values dominate the scale.

A transformation such as:

$$
x'=\log(x)
$$

compresses the range.

This can make a highly skewed feature easier for some models to work with.

But:

> **EDA identifies the skew; preprocessing/feature engineering decides whether and how to transform it.**

---

# 29. Box Plot

```python
df["salary"].plot(kind="box")
```

A box plot summarizes:

```text
minimum-ish
Q1
median
Q3
maximum-ish
potential outliers
```

Conceptually:

```text
       •
       •      ← possible outliers
       │
   ┌─────────┐
───│    │    │───
   └─────────┘
       │
```

---

# 30. The IQR Outlier Rule

A common statistical rule:

$$
Lower=Q_1-1.5(IQR)
$$

$$
Upper=Q_3+1.5(IQR)
$$

Values outside these bounds are flagged as potential outliers.

Important word:

> **Potential**

An outlier is not automatically an error.

---

# 31. Outlier ≠ Bad Data

Suppose:

```text
ages:

21
22
23
24
25
95
```

95 is unusual.

But maybe the customer really is 95.

So:

```text
Rare ≠ wrong
```

Another example:

A particle detector records an extremely high energy event.

That might be:

```text
instrument error
```

or:

```text
genuine scientific discovery
```

You investigate.

You don't automatically delete it.

---

# 32. Categorical Univariate Analysis

For categories:

```python
df["city"].value_counts()
```

You might get:

```text
Karachi      50,000
Lahore       30,000
Islamabad    15,000
Quetta        5,000
```

Visualize:

```python
df["city"].value_counts().plot(kind="bar")
```

Now you can detect:

* dominant categories
* rare categories
* imbalance
* unexpected values
* spelling problems

---

# 33. Bivariate Analysis

**Bi = two variables.**

Now we're asking:

> **How does one variable relate to another?**

Examples:

```text
age ↔ salary
distance ↔ delivery_time
study_hours ↔ exam_score
temperature ↔ energy_consumption
```

This is where EDA becomes particularly useful for feature engineering.

---

# 34. Scatter Plot

For two numerical variables:

```python
df.plot(
    x="study_hours",
    y="exam_score",
    kind="scatter"
)
```

Suppose we see:

```text
exam score
100 |                 •
 90 |             • •
 80 |          • •
 70 |       • •
 60 |    •
    └────────────────────
       study hours
```

There appears to be a positive relationship.

More study → generally higher score.

---

# 35. Correlation

A common measure is Pearson correlation:

$$
r =
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
$$

It ranges from:

$$
-1 \le r \le 1
$$

Interpretation:

```text
r ≈ +1
Strong positive linear relationship

r ≈ 0
Little/no linear relationship

r ≈ -1
Strong negative linear relationship
```

---

# 36. But Correlation Is NOT Causation 🚨

Suppose:

```text
Ice cream sales ↑
Drowning incidents ↑
```

They correlate.

Does ice cream cause drowning?

No.

A third variable:

```text
temperature
```

can influence both.

This is a classic interview question.

**Answer:**

> Correlation measures statistical association, while causation means a change in one variable directly produces a change in another under a causal interpretation. Correlation alone does not establish causality.

---

# 37. Pearson vs Spearman

At intermediate level, know the distinction.

### Pearson

Measures **linear** association.

### Spearman

Measures **monotonic** association using ranks.

Example:

```text
X: 1 2 3 4 5
Y: 1 4 9 16 25
```

The relationship is nonlinear but monotonic.

Pearson and Spearman can behave differently.

Mental model:

```text
Pearson  → "Are they linearly related?"
Spearman → "Do they generally move together in order?"
```

---

# 38. Correlation Matrix

```python
df.corr(numeric_only=True)
```

Might produce:

|          |  age | income | spending | score |
| -------- | ---: | -----: | -------: | ----: |
| age      | 1.00 |    .65 |      .20 |   .10 |
| income   |  .65 |   1.00 |      .72 |   .31 |
| spending |  .20 |    .72 |     1.00 |   .45 |
| score    |  .10 |    .31 |      .45 |  1.00 |

This helps identify relationships.

You can visualize it as a heatmap.

But remember:

> Correlation is only one type of relationship.

A correlation matrix can miss important **nonlinear** relationships.

---

# 39. The Famous Trap: Correlation ≈ 0 Doesn't Mean "No Relationship"

Consider:

$$
y=x^2
$$

Suppose:

```text
x = -3,-2,-1,0,1,2,3
```

There is a strong relationship.

But the positive and negative sides can cancel out in Pearson correlation.

So:

```text
correlation ≈ 0
```

doesn't necessarily mean:

```text
no predictive relationship
```

This is why visualization matters.

---

# 40. Categorical vs Numerical

Suppose:

```text
city → salary
```

One is categorical, one numerical.

You can investigate:

```python
df.groupby("city")["salary"].mean()
```

But don't stop at mean.

Also consider:

```text
median
distribution
sample size
variance
```

A category with 3 observations shouldn't necessarily be treated the same as one with 30,000 observations.

---

# 41. Numerical vs Categorical Target

Suppose classification:

```text
cancelled = 0/1
```

You can ask:

> Does delivery time differ between cancelled and non-cancelled orders?

For example:

```python
df.groupby("cancelled")["delivery_time"].mean()
```

Maybe:

```text
cancelled = 0 → 28 min
cancelled = 1 → 52 min
```

That's a potentially useful signal.

---

# 42. Multivariate Analysis

**Multi = more than two variables.**

Real-world ML is multivariate.

For example:

```text
delivery_time
    ↑
    │
    ├── distance
    ├── traffic
    ├── weather
    ├── restaurant_load
    ├── hour
    └── driver_availability
```

A variable might look weak individually but become important when combined with others.

---

# 43. Simpson's Paradox — Intermediate-Level EDA

This is a fascinating concept.

Suppose overall:

```text
Model A seems better than Model B
```

But when you split by groups:

```text
Group 1 → B better
Group 2 → B better
Group 3 → B better
```

How?

Because group proportions differ.

This teaches an important EDA lesson:

> **Aggregated data can hide patterns present within subgroups.**

Therefore investigate important segments.

Examples:

```text
overall
   ↓
by city
by gender
by age group
by device
by customer type
by time period
```

when scientifically/business appropriate.

---

# 44. Target Variable Analysis 🎯

This is one of the most important parts of EDA.

Your target is:

```text
y
```

Before training, understand it.

---

## Regression

Suppose:

```text
house_price
```

Investigate:

```python
df["house_price"].describe()
```

Ask:

```text
Is it skewed?
Are there extreme values?
Are there impossible prices?
Is the target highly concentrated?
```

---

## Classification

Suppose:

```text
fraud = 0/1
```

Check:

```python
df["fraud"].value_counts()
```

Maybe:

```text
0 → 990,000
1 → 10,000
```

That's:

```text
99% non-fraud
1% fraud
```

Severe class imbalance.

---

# 45. Why Class Imbalance Matters

Imagine a fraud model that predicts:

```text
EVERY transaction = not fraud
```

Accuracy:

$$
99\%
$$

Sounds fantastic.

But fraud detection performance:

```text
Terrible
```

because it catches:

```text
0% of fraud
```

This is why evaluation metrics must reflect the problem.

And the issue should already be discovered during EDA.

---

# 46. Time Is a Special Dimension ⏰

This is a major intermediate ML concept.

Suppose you're predicting:

> tomorrow's sales.

Your data:

```text
2024
2025
2026
```

Don't randomly assume:

```python
train_test_split(...)
```

is always appropriate.

Why?

Because future information must not leak into the past.

A more realistic setup might be:

```text
TRAIN
2024 ───────── 2025

VALIDATION
early 2026

TEST
later 2026
```

EDA should investigate:

```text
data over time
```

Look for:

* trends
* seasonality
* sudden changes
* missing periods
* distribution shifts
* future leakage

---

# 47. Data Leakage — EDA's Most Dangerous Enemy 🚨

Imagine predicting whether a patient will be admitted.

You have:

```text
age
temperature
blood_pressure
symptoms
```

Good.

But you also have:

```text
discharge_status
```

That information becomes available **after hospitalization**.

If you're predicting admission before it happens, that's leakage.

The model might achieve:

```text
99.9% accuracy
```

but the model is useless in reality.

---

# 48. The Golden Leakage Question

For every feature ask:

> **"Would this information genuinely be available at prediction time?"**

This question should become automatic.

For example:

### Predict house price today

Can you use:

```text
sale_price
```

?

No.

Can you use:

```text
number_of_rooms
area
location
```

?

Yes.

---

# 49. Train/Test Distribution Analysis

Another intermediate-level skill.

Suppose training data:

```text
Age:
20–60
```

But test data:

```text
Age:
20–95
```

Your model hasn't really seen older people during training.

That's a potential distribution-shift problem.

You should compare:

```text
P_train(X)
```

with:

```text
P_test(X)
```

Conceptually:

```text
TRAIN DISTRIBUTION
       ↓
       ███████
    ███████████

TEST DISTRIBUTION
       ↓
    █████████
  █████████████
```

Large differences can indicate **distribution shift**.

---

# 50. Train/Test Leakage Through EDA

Be careful here.

You can inspect train and test distributions to understand whether they differ.

But don't use test labels to improve your model.

And don't use information from the test set to fit preprocessing transformations.

Remember:

```text
Training data
   ↓
Learn

Validation
   ↓
Choose/tune

Test
   ↓
Final evaluation
```

---

# 51. EDA and Feature Engineering Are Connected

Suppose EDA reveals:

```text
delivery_time strongly increases
during peak hours
```

You might create:

```text
is_peak_hour
```

EDA:

> discovers pattern.

Feature engineering:

> converts useful domain knowledge/pattern into a model input.

This distinction is important.

---

# 52. EDA → Hypothesis → Test

Intermediate ML engineers shouldn't just generate 50 plots.

Instead:

```text
Observation
    ↓
Hypothesis
    ↓
Analysis
    ↓
Evidence
    ↓
Decision
```

Example:

### Observation

Longer delivery times appear associated with cancellations.

### Hypothesis

> Orders with longer delivery times are more likely to be cancelled.

### Analysis

Compare:

```text
delivery_time
```

against:

```text
cancelled
```

### Decision

If the relationship is meaningful and survives proper validation:

```text
delivery_time
```

may be useful.

That's **thinking**, rather than plot collecting.

---

# 53. EDA Is NOT Just Visualization

This is a common interview question.

### Bad answer:

> "EDA means making graphs."

### Better answer:

> **EDA is the systematic process of understanding dataset structure, quality, distributions, relationships, anomalies, target behavior, and potential leakage using statistical analysis, visualization, and domain knowledge before modeling.**

Visualization is a tool inside EDA.

Not EDA itself.

---

# 54. A Professional EDA Checklist

When you receive a new dataset, run through this.

## A. Structure

```python
df.shape
df.head()
df.tail()
df.columns
df.info()
df.dtypes
```

Ask:

* How many rows?
* How many columns?
* What are the data types?
* Which column is the target?
* What does each column mean?

---

## B. Data Quality

```python
df.isna().sum()
df.duplicated().sum()
```

Check:

* missing values
* duplicates
* invalid values
* inconsistent categories
* wrong units
* incorrect data types

---

## C. Numerical Features

```python
df.describe()
```

Investigate:

* mean
* median
* std
* min/max
* quantiles
* skewness
* outliers

---

## D. Categorical Features

```python
df["city"].value_counts()
```

Investigate:

* unique categories
* rare categories
* dominant categories
* inconsistent spelling
* unexpected categories

---

## E. Target

Ask:

```text
What does y look like?
```

For classification:

```text
class balance?
```

For regression:

```text
distribution?
skew?
outliers?
```

---

## F. Relationships

Investigate:

```text
feature ↔ feature
feature ↔ target
```

Use:

* scatter plots
* correlation
* groupby
* box plots
* distributions
* contingency tables

---

## G. Leakage

For every feature:

> Could I know this at prediction time?

---

## H. Distribution Shift

Compare:

```text
training
vs
validation
vs
test
```

when those splits are available.

---

# 55. A Practical EDA Template

A basic starting notebook might look like:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# ------------------
# 1. Basic structure
# ------------------

print(df.shape)
print(df.columns)
print(df.info())

display(df.head())
display(df.tail())

# ------------------
# 2. Statistics
# ------------------

display(df.describe())

# ------------------
# 3. Missing values
# ------------------

print(df.isnull().sum())

# ------------------
# 4. Duplicates
# ------------------

print("Duplicates:", df.duplicated().sum())

# ------------------
# 5. Categorical
# ------------------

for col in df.select_dtypes(include="object"):
    print("\n", col)
    print(df[col].value_counts().head(20))

# ------------------
# 6. Numerical
# ------------------

numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:
    df[col].hist()
    plt.title(col)
    plt.show()

# ------------------
# 7. Correlation
# ------------------

corr = df[numeric_cols].corr()
display(corr)
```

This is a **starting template**, not a complete professional EDA.

---

# 56. Don't Build a "50-Plot Monster" 😭

A beginner often does:

```text
20 histograms
10 scatterplots
5 heatmaps
15 boxplots
```

and then says:

> "I completed EDA."

Not necessarily.

Professional EDA asks:

> **What question does this analysis answer?**

Every analysis should have a purpose.

---

# 57. EDA in a Real ML Project

Let's put everything together.

Suppose you're building:

> **Taxi fare prediction**

Dataset:

```text
pickup_time
dropoff_time
distance
passenger_count
weather
traffic
fare
```

### Step 1

Inspect:

```text
1,000,000 rows
```

### Step 2

Discover:

```text
distance has 500 negative values
```

🚨 Data problem.

### Step 3

Discover:

```text
passenger_count max = 78
```

Potentially invalid.

### Step 4

Discover:

```text
fare is heavily right-skewed
```

Potential transformation/robust modeling consideration.

### Step 5

Discover:

```text
distance strongly related to fare
```

Makes domain sense.

### Step 6

Discover:

```text
late-night trips have higher average fares
```

Potential feature:

```text
hour
```

### Step 7

Discover:

```text
some records have dropoff_time before pickup_time
```

🚨 Invalid data.

### Step 8

Discover:

```text
dropoff_time
```

was accidentally generated after the fare was finalized, while the prediction is supposed to happen at pickup.

Potential leakage depending on the exact prediction task.

Now EDA has actually **changed the modeling strategy**.

That's what good EDA does.

---

# 58. EDA for Your Physics Background ⚛️

This is where you have an advantage.

Imagine experimental data:

```text
time
temperature
pressure
magnetic_field
voltage
current
```

You're investigating:

$$
I = f(V,T,B)
$$

EDA asks:

```text
Are measurements physically plausible?
Are there sensor spikes?
Is noise Gaussian?
Does variance change with temperature?
Are there calibration shifts?
Are measurements correlated?
Is there drift over time?
Are there missing intervals?
Are there regime changes?
```

You can use domain laws as sanity checks.

For example:

$$
V = IR
$$

If your measurements imply something wildly inconsistent with known physical constraints, EDA can catch it.

This is actually one of the strengths you can bring as a physics-trained ML engineer:

> **You don't have to treat data as arbitrary numbers.**

---

# 59. EDA for Computer Vision

Since you're also interested in computer vision, EDA looks different.

Instead of:

```text
age
salary
distance
```

you might investigate:

```text
image dimensions
class distribution
brightness
contrast
blur
duplicate images
corrupted images
annotation quality
bounding-box sizes
bounding-box locations
class imbalance
train/validation/test leakage
```

For object detection:

```text
Images
   ↓
Annotations
   ↓
Class distribution
   ↓
Boxes per image
   ↓
Box sizes
   ↓
Box locations
   ↓
Small-object frequency
   ↓
Image quality
   ↓
Duplicate/near-duplicate images
```

So EDA is **domain-specific**.

---

# 60. EDA for Your Quantum ML Work ⚛️🤖

Suppose your dataset contains:

```text
temperature
noise_strength
T1
T2
gate_time
frequency
decoherence_rate
fidelity
```

EDA could reveal:

```text
noise_strength ↑
        ↓
fidelity ↓
```

or:

```text
temperature ↑
        ↓
T1 ↓
```

You can investigate whether relationships are:

* linear
* nonlinear
* regime-dependent
* noisy
* correlated
* physically plausible

You can then create physically meaningful features.

That's where:

**Physics → EDA → Feature Engineering → ML**

becomes very powerful.

---

# 61. ⭐ Intermediate ML Engineer EDA Skills

If you want to move beyond beginner level, learn these deeply:

### Statistical understanding

* mean
* median
* variance
* standard deviation
* percentiles
* IQR
* covariance
* correlation
* skewness
* distributions

### Visualization

* histogram
* box plot
* scatter plot
* bar plot
* line plot
* heatmap
* pair plot

### Data quality

* missingness
* duplicates
* invalid values
* inconsistent categories
* outliers
* data types
* units

### ML-specific EDA

* target distribution
* class imbalance
* feature-target relationships
* feature-feature relationships
* leakage
* train/test distribution
* temporal leakage
* distribution shift

### Deeper thinking

* correlation ≠ causation
* Simpson's paradox
* confounding
* sampling bias
* selection bias
* measurement bias
* survivorship bias

These last concepts are where EDA starts becoming **data science rather than just Pandas**.

---

# 🎯 Interview Questions & Answers

Now let's train your interview brain.

---

## Q1. What is EDA?

**Answer:**

> Exploratory Data Analysis is the systematic process of understanding a dataset's structure, quality, distributions, relationships, anomalies, target behavior, and potential issues such as missing values, outliers, imbalance, and leakage using statistical techniques, visualization, and domain knowledge.

---

## Q2. Why is EDA important before ML?

**Answer:**

> EDA helps identify data quality problems, understand feature and target distributions, discover relationships, detect outliers and class imbalance, uncover leakage, and guide preprocessing, feature engineering, model selection, and evaluation.

---

## Q3. What is the difference between EDA and data cleaning?

**Answer:**

> EDA is primarily about investigating and understanding the data, while data cleaning is about correcting or handling identified problems such as missing values, duplicates, invalid values, and inconsistent formats.

Simple:

```text
EDA       → Discover
Cleaning  → Fix
```

---

## Q4. What is an outlier?

**Answer:**

> An outlier is an observation that is unusually distant from the majority of observations according to a chosen statistical or domain criterion.

Important:

> An outlier isn't necessarily an error.

---

## Q5. How do you detect outliers?

Possible methods:

* IQR rule
* Z-score
* box plots
* distribution plots
* robust statistics
* domain constraints
* model-based methods

Don't say:

> "I delete values above 3 standard deviations."

That's too simplistic.

---

## Q6. Mean vs median?

**Answer:**

> Mean uses every value and is sensitive to extreme observations, while median is the middle value after sorting and is generally more robust to outliers and skewed distributions.

---

## Q7. What does correlation measure?

**Answer:**

> Correlation measures the strength and direction of statistical association between variables. Pearson correlation specifically measures linear association.

---

## Q8. Can correlation prove causation?

**Answer:**

> No. Correlation indicates association but does not establish that one variable causes another. Confounding variables, reverse causality, selection effects, and other factors can produce correlations.

---

## Q9. What does correlation = 0 mean?

**Good answer:**

> For Pearson correlation, it means there is no linear correlation, but a nonlinear relationship may still exist.

Excellent interview answer.

---

## Q10. What is class imbalance?

**Answer:**

> Class imbalance occurs when some classes have substantially more observations than others.

Example:

```text
Normal → 99%
Fraud  → 1%
```

Accuracy may therefore be misleading.

---

## Q11. How would you handle class imbalance?

Don't immediately say:

> "Use SMOTE."

A stronger answer:

> First I would understand the business/scientific cost of false positives and false negatives. I would then consider appropriate metrics such as precision, recall, F1 or PR-AUC, class weighting, resampling techniques such as oversampling/undersampling, threshold adjustment, or specialized approaches depending on the problem.

---

## Q12. What is data leakage?

**Answer:**

> Data leakage occurs when information that would not legitimately be available when making a prediction is used during model training, causing unrealistically good validation or test performance.

---

## Q13. How do you detect leakage?

Ask:

> **Could this feature be known at prediction time?**

Also investigate:

* suspiciously high performance
* features created after the target event
* duplicated information
* preprocessing performed before splitting
* future information
* target-derived features

---

## Q14. Why shouldn't you remove every outlier?

**Answer:**

> Because an outlier may represent a legitimate rare observation rather than an error. Removing it without understanding its origin can destroy valuable information or bias the dataset.

---

## Q15. What is skewness?

**Answer:**

> Skewness describes asymmetry in a distribution. Positive skew usually has a longer right tail, while negative skew has a longer left tail.

---

## Q16. Why might you use log transformation?

**Answer:**

> Log transformation can reduce strong right skew, compress extreme values, stabilize scale, and sometimes make relationships easier for certain models to learn.

---

## Q17. What is multicollinearity?

Suppose:

```text
house_area
house_size_sqft
house_area_m2
```

are highly related.

Your features contain redundant information.

This is **multicollinearity**.

It can particularly cause problems for models whose coefficients are interpreted directly, such as linear regression.

Tree models are generally less sensitive to multicollinearity in terms of predictive performance, though redundant features can still have consequences for interpretation and efficiency.

---

## Q18. How do you detect multicollinearity?

Common approaches:

```text
correlation matrix
VIF
```

VIF:

$$
VIF_j=\frac{1}{1-R_j^2}
$$

A high VIF indicates that a feature can be strongly explained by other predictors.

---

# 🧠 The EDA Mental Model You Should Never Forget

Imagine you are handed a mysterious box.

Inside:

```text
RAW DATA
```

You don't build the robot immediately.

First:

```text
             🔎 EDA DETECTIVE
                    │
                    ↓
          "WHAT IS INSIDE?"
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      STRUCTURE             QUALITY
          │                   │
      columns              missing
      dtypes               duplicates
      shape                invalid
          │                   │
          └─────────┬─────────┘
                    ↓
              DISTRIBUTIONS
                    │
                    ↓
               RELATIONSHIPS
                    │
                    ↓
             TARGET BEHAVIOR
                    │
                    ↓
              OUTLIERS?
                    │
                    ↓
              IMBALANCE?
                    │
                    ↓
               LEAKAGE?
                    │
                    ↓
           DISTRIBUTION SHIFT?
                    │
                    ↓
            ┌───────────────┐
            │ EDA CONCLUSION│
            └───────┬───────┘
                    ↓
        Cleaning / Preprocessing
                    ↓
          Feature Engineering
                    ↓
              ML Modeling
```

## The single most important idea

> **EDA isn't about making pretty graphs. EDA is about asking the data questions before allowing an ML model to make assumptions about it.**

And for an intermediate ML engineer, your goal is to move from:

> **"I know how to run `describe()`."**

to:

> **"I can receive an unfamiliar dataset, investigate its structure and statistical behavior, identify data-quality and ML risks, formulate hypotheses, validate them, and translate my findings into modeling decisions."**

That is the real EDA skill.
