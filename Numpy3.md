# NumPy — Part 3: Operations, Statistics, Boolean Masking & Broadcasting

Now we're getting into the **real power of NumPy**.

In Part 2, you learned how to *look inside* arrays:

```text
array → shape → rows → columns → indexing → slicing
```

Now we'll learn how to **do mathematics on the entire dataset at once**.

And this is where NumPy starts becoming directly connected to ML.

---

# 1. The Big Idea

Suppose you have:

```python
import numpy as np

scores = np.array([50, 60, 70, 80, 90])
```

You want to add 5 to every score.

With normal Python, you might think:

```python
for i in range(len(scores)):
    scores[i] += 5
```

NumPy lets you simply write:

```python
scores + 5
```

Result:

```text
[55 65 75 85 95]
```

This is called **vectorized computation**.

> Instead of telling Python *how to process each number*, you tell NumPy *what mathematical operation you want*.

This idea will appear everywhere in ML.

---

# 2. Arithmetic Operations

Let's start simple.

```python
x = np.array([10, 20, 30, 40])
```

### Addition

```python
x + 5
```

```text
[15 25 35 45]
```

### Subtraction

```python
x - 5
```

```text
[ 5 15 25 35]
```

### Multiplication

```python
x * 2
```

```text
[20 40 60 80]
```

### Division

```python
x / 10
```

```text
[1. 2. 3. 4.]
```

### Power

```python
x ** 2
```

```text
[ 100  400  900 1600]
```

---

# 3. Think Mathematically

When you write:

```python
x * 2
```

and:

```text
x = [10, 20, 30, 40]
```

NumPy is conceptually doing:

```text
[10, 20, 30, 40]
       × 2

[10×2, 20×2, 30×2, 40×2]

= [20, 40, 60, 80]
```

This is **element-wise operation**.

---

# 4. Array + Array

Suppose:

```python
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
```

Then:

```python
a + b
```

gives:

```text
[11 22 33]
```

Because:

```text
10 + 1 = 11
20 + 2 = 22
30 + 3 = 33
```

Similarly:

```python
a - b
```

```text
[ 9 18 27]
```

and:

```python
a * b
```

```text
[10 40 90]
```

---

# 5. Important Distinction: Element-wise vs Matrix Multiplication

This is **very important for ML**.

If:

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])
```

Then:

```python
A * B
```

means:

```text
1×5   2×6
3×7   4×8
```

giving:

```text
[[ 5 12]
 [21 32]]
```

But:

```python
A @ B
```

means **matrix multiplication**:

```text
[[19 22]
 [43 50]]
```

We'll spend much more time on `@` when we reach NumPy's linear algebra section.

---

# 6. Comparison Operations

NumPy can compare every element simultaneously.

```python
scores = np.array([45, 60, 75, 90, 55])
```

Ask:

```python
scores > 60
```

Result:

```text
[False False  True  True False]
```

NumPy checked:

```text
45 > 60 → False
60 > 60 → False
75 > 60 → True
90 > 60 → True
55 > 60 → False
```

This is incredibly useful.

---

# 7. Other Comparisons

```python
scores >= 60
```

```text
[False True True True False]
```

Because 60 itself qualifies.

Similarly:

```python
scores < 60
```

```text
[ True False False False  True]
```

And:

```python
scores == 60
```

```text
[False True False False False]
```

---

# 8. Boolean Masking

Now something very powerful.

We have:

```python
scores = np.array([45, 60, 75, 90, 55])
```

First:

```python
scores > 60
```

gives:

```text
[False False True True False]
```

Now put that inside the array:

```python
scores[scores > 60]
```

Result:

```text
[75 90]
```

This means:

> Give me only the values for which the condition is true.

This is called **Boolean masking**.

---

# 9. Real-World Example

Imagine employee salaries:

```python
salary = np.array([
    35000,
    50000,
    75000,
    120000,
    45000,
    150000
])
```

Find employees earning more than 100,000:

```python
salary[salary > 100000]
```

Result:

```text
[120000 150000]
```

This is much closer to how you'll actually use NumPy.

---

# 10. Count How Many Satisfy a Condition

Suppose:

```python
scores = np.array([45, 60, 75, 90, 55])
```

We want the number of students who scored above 60.

First:

```python
scores > 60
```

gives:

```text
[False False True True False]
```

Now:

```python
np.sum(scores > 60)
```

Result:

```text
2
```

Why does this work?

Because NumPy treats:

```text
True  → 1
False → 0
```

So:

```text
0 + 0 + 1 + 1 + 0 = 2
```

Very useful trick.

---

# 11. `np.where()`

Another useful operation is `np.where()`.

Suppose:

```python
scores = np.array([45, 60, 75, 90, 55])
```

We want to label students:

```text
score >= 60 → Pass
score < 60  → Fail
```

We can use:

```python
np.where(scores >= 60, "Pass", "Fail")
```

Result:

```text
['Fail' 'Pass' 'Pass' 'Pass' 'Fail']
```

Conceptually:

```text
np.where(condition, value_if_true, value_if_false)
```

---

# 12. `np.where()` for Numerical Data

Suppose we want to give a bonus:

```text
salary > 100000 → 10000 bonus
otherwise       → 0
```

```python
bonus = np.where(
    salary > 100000,
    10000,
    0
)
```

This is another example of vectorized thinking.

---

# 13. Statistical Operations

Now let's move toward ML.

Consider:

```python
scores = np.array([50, 60, 70, 80, 90])
```

### Mean

```python
np.mean(scores)
```

Result:

```text
70
```

Mathematically:

```text
(50 + 60 + 70 + 80 + 90) / 5
= 70
```

---

# 14. Shortcut Syntax

Instead of:

```python
np.mean(scores)
```

you can also write:

```python
scores.mean()
```

Both are commonly used.

Likewise:

```python
scores.sum()
scores.min()
scores.max()
scores.std()
scores.var()
```

---

# 15. Sum

```python
scores.sum()
```

Result:

```text
350
```

---

# 16. Minimum and Maximum

```python
scores.min()
```

→ `50`

```python
scores.max()
```

→ `90`

You can also find their positions.

```python
scores.argmin()
```

→ `0`

```python
scores.argmax()
```

→ `4`

So:

> `argmax()` tells you **where** the maximum occurs.

This distinction is important.

```text
max()     → maximum value
argmax()  → index of maximum value
```

---

# 17. Standard Deviation

Standard deviation tells us roughly:

> How spread out are the values around their mean?

For:

```python
scores = np.array([50, 60, 70, 80, 90])
```

we can calculate:

```python
scores.std()
```

The exact number isn't the important part right now.

The concept is:

```text
small std → values are close together
large std → values are spread out
```

Standard deviation becomes **extremely important when we study feature standardization**.

---

# 18. Variance

Variance is:

```text
variance = standard_deviation²
```

NumPy:

```python
scores.var()
```

You don't need to memorize the mathematical derivation yet.

Just understand:

```text
std → spread
variance → squared measure of spread
```

---

# 19. Real ML Dataset

Let's create something closer to a real ML dataset.

```python
X = np.array([
    [20, 40000, 2],
    [25, 50000, 3],
    [30, 70000, 5],
    [35, 90000, 8],
    [40, 120000, 12]
])
```

Suppose the columns are:

```text
column 0 → Age
column 1 → Salary
column 2 → Experience
```

And:

```python
X.shape
```

gives:

```text
(5, 3)
```

So:

```text
5 samples
3 features
```

---

# 20. Overall Mean

If you do:

```python
X.mean()
```

NumPy considers **all values**.

But that's often not what we want in ML.

We usually want:

> Mean of each feature.

That's where `axis` becomes critical.

---

# 21. `axis=0`

Run:

```python
X.mean(axis=0)
```

Conceptually:

```text
Age column       → average age
Salary column    → average salary
Experience       → average experience
```

So you get something like:

```text
[30, 74000, 6]
```

The exact values depend on the data.

The important idea:

```text
axis=0 → reduce down rows → result for each column
```

---

# 22. Visualizing `axis=0`

Take:

```text
        Age   Salary   Exp
         ↓      ↓      ↓
        20     40000    2
        25     50000    3
        30     70000    5
        35     90000    8
        40    120000   12
```

`axis=0` means:

```text
↓      ↓      ↓
↓      ↓      ↓
↓      ↓      ↓
```

So you calculate:

```text
mean age
mean salary
mean experience
```

---

# 23. `axis=1`

Now:

```python
X.mean(axis=1)
```

means:

> Calculate the mean of each row.

So:

```text
Employee 1 → average of [20, 40000, 2]
Employee 2 → average of [25, 50000, 3]
...
```

This particular calculation isn't very meaningful because age, salary and experience have different units, but it demonstrates the mechanics.

For ML, `axis=1` often means:

> Perform an operation across the features of each sample.

---

# 24. The Axis Rule

For a 2D array:

```text
axis=0 → operate vertically → one result per column
axis=1 → operate horizontally → one result per row
```

Memorize the **result**, not just the wording.

```python
X.mean(axis=0)
```

→ one mean per feature.

```python
X.mean(axis=1)
```

→ one mean per sample.

---

# 25. Very Important ML Example: Standardization

Suppose:

```python
X = np.array([
    [20, 40000],
    [25, 50000],
    [30, 70000],
    [35, 90000],
    [40, 120000]
])
```

We have:

```text
Age       Salary
20        40000
25        50000
30        70000
35        90000
40        120000
```

Notice:

```text
Age     → tens
Salary  → tens of thousands
```

Their scales are very different.

We calculate feature means:

```python
mean = X.mean(axis=0)
```

Then standard deviations:

```python
std = X.std(axis=0)
```

Then:

```python
X_scaled = (X - mean) / std
```

This is standardization.

Don't worry if this looks complicated right now.

You've actually already learned almost everything required to understand this expression:

```text
mean
axis
subtraction
division
broadcasting
```

---

# 26. Broadcasting

Now let's understand the magic that makes the previous operation work.

Suppose:

```python
X = np.array([
    [10, 20],
    [30, 40],
    [50, 60]
])
```

and:

```python
mean = np.array([30, 40])
```

Then:

```python
X - mean
```

works.

Conceptually NumPy behaves like:

```text
10 20      30 40
30 40  -   30 40
50 60      30 40
```

giving:

```text
-20 -20
  0   0
 20  20
```

NumPy **broadcasts** the smaller array across the larger one.

---

# 27. Broadcasting: Simple Example

```python
x = np.array([10, 20, 30])
```

Then:

```python
x + 5
```

is conceptually:

```text
10 20 30
+5 +5 +5
---------
15 25 35
```

You didn't manually create:

```python
[5, 5, 5]
```

NumPy handles the broadcasting.

---

# 28. Broadcasting with Different Shapes

This is where broadcasting becomes more interesting.

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

Shape:

```text
(2, 3)
```

And:

```python
v = np.array([10, 20, 30])
```

Shape:

```text
(3,)
```

Then:

```python
X + v
```

works:

```text
1 2 3       10 20 30
4 5 6   +   10 20 30

11 22 33
14 25 36
```

Why?

Because:

```text
(2, 3)
(3,)
```

are compatible.

The vector is broadcast across the rows.

---

# 29. Broadcasting Mental Model

Think:

```text
Large array
      +
Small compatible array
      ↓
Small array is conceptually repeated
      ↓
Operation happens element-wise
```

This is one of the reasons NumPy can express complicated mathematical operations in just one line.

---

# 30. Boolean Masking + 2D Data

Now let's combine what we know.

Suppose:

```python
students = np.array([
    [18, 5, 80],
    [19, 7, 90],
    [20, 3, 65],
    [18, 8, 95],
    [21, 4, 70]
])
```

Columns:

```text
Age
Study Hours
Exam Score
```

We want students who scored above 80.

First:

```python
students[:, 2] > 80
```

gives:

```text
[False True False True False]
```

Now:

```python
students[students[:, 2] > 80]
```

Result:

```text
[[19  7 90]
 [18  8 95]]
```

This is a **very important ML/data-processing pattern**.

---

# 31. Filtering Based on One Feature

Suppose we want students who studied more than 5 hours:

```python
students[students[:, 1] > 5]
```

Result:

```text
[[19  7 90]
 [18  8 95]]
```

Notice the pattern:

```python
X[X[:, feature_column] > condition]
```

You will see patterns like this frequently when working with data.

---

# 32. Multiple Conditions

Suppose we want:

```text
Study Hours > 5
AND
Score > 80
```

Write:

```python
students[
    (students[:, 1] > 5) &
    (students[:, 2] > 80)
]
```

Result:

```text
[[19  7 90]
 [18  8 95]]
```

Remember:

```text
& → AND
| → OR
```

And put each condition inside parentheses.

---

# 33. OR

Suppose we want:

```text
study hours > 7
OR
score > 90
```

```python
students[
    (students[:, 1] > 7) |
    (students[:, 2] > 90)
]
```

---

# 34. NOT

Suppose:

```python
scores = np.array([40, 60, 70, 90])
```

We can create:

```python
scores >= 60
```

→

```text
[False True True True]
```

Then:

```python
~(scores >= 60)
```

→

```text
[True False False False]
```

`~` means logical NOT.

---

# 35. `any()` and `all()`

These are useful when working with conditions.

Suppose:

```python
scores = np.array([70, 80, 90])
```

Check:

```python
scores > 85
```

→

```text
[False False True]
```

Does **any** student have a score above 85?

```python
np.any(scores > 85)
```

→

```text
True
```

Does **every** student have a score above 85?

```python
np.all(scores > 85)
```

→

```text
False
```

---

# 36. Why This Matters in ML

Imagine you're checking model predictions.

You might want to know:

```text
Are any predictions negative?
Are all probabilities between 0 and 1?
Did every sample receive a prediction?
```

NumPy's:

```python
np.any()
np.all()
```

can help answer these questions efficiently.

---

# 37. Useful Mathematical Functions

NumPy has many mathematical functions.

```python
x = np.array([1, 4, 9, 16])
```

Square root:

```python
np.sqrt(x)
```

→

```text
[1. 2. 3. 4.]
```

Absolute value:

```python
np.abs(np.array([-5, -2, 3]))
```

→

```text
[5 2 3]
```

Exponential:

```python
np.exp(x)
```

Log:

```python
np.log(x)
```

These become particularly important when we study:

* probability
* loss functions
* activation functions
* normalization
* statistical models

---

# 38. A Very Important ML Function: `log`

You may remember from your previous ML learning that we sometimes transform a variable using:

```text
log(x)
```

For example:

```python
salary = np.array([
    30000,
    50000,
    100000,
    500000
])
```

We can calculate:

```python
np.log(salary)
```

The log compresses large numerical ranges.

That's why transformations such as:

```python
log_salary = np.log(salary)
```

can be useful in feature engineering.

---

# 39. Another Important Function: `clip`

Suppose:

```python
x = np.array([-10, 5, 20, 100])
```

We want every value constrained between 0 and 50.

```python
np.clip(x, 0, 50)
```

Result:

```text
[ 0  5 20 50]
```

Conceptually:

```text
below 0  → becomes 0
0-50     → unchanged
above 50 → becomes 50
```

This can be useful for controlling extreme values.

---

# 40. Sorting

```python
scores = np.array([80, 45, 90, 60, 70])
```

Sort:

```python
np.sort(scores)
```

Result:

```text
[45 60 70 80 90]
```

Find sorting indices:

```python
np.argsort(scores)
```

This tells you the positions that would sort the array.

This is useful when ranking predictions, scores, distances, etc.

---

# 41. Random Data

ML frequently needs random data.

Use NumPy's modern random generator:

```python
rng = np.random.default_rng(42)
```

Now:

```python
rng.random(5)
```

generates 5 random numbers between 0 and 1.

The `42` is a **seed**.

---

# 42. Why Seeds Matter

Suppose you're debugging your ML model.

If your data changes randomly every time, it becomes difficult to reproduce the result.

So:

```python
rng = np.random.default_rng(42)
```

lets you generate reproducible random numbers.

Same seed:

```text
42
```

→ same sequence.

Different seed:

```text
100
```

→ different sequence.

This idea becomes important when training ML models.

---

# 43. Random Integers

```python
rng.integers(1, 100, size=10)
```

This generates 10 random integers in the specified range.

You can use this to create toy datasets.

For example:

```python
rng = np.random.default_rng(42)

ages = rng.integers(18, 60, size=100)
```

Now you have 100 simulated ages.

---

# 44. Random Normal Distribution

You can generate normally distributed data:

```python
rng.normal(
    loc=0,
    scale=1,
    size=1000
)
```

Here:

```text
loc   → mean
scale → standard deviation
size  → number of values
```

This becomes important when studying:

* Gaussian distributions
* noise
* simulations
* ML initialization
* statistics

And given your physics background, you'll encounter this kind of numerical simulation often.

---

# 45. A Mini ML Exercise

Let's combine everything.

```python
students = np.array([
    [18, 5, 80],
    [19, 7, 90],
    [20, 3, 65],
    [18, 8, 95],
    [21, 4, 70]
])
```

Remember:

```text
column 0 → Age
column 1 → Study Hours
column 2 → Score
```

### Question 1

Find the average score.

Think:

```python
?
```

Answer:

```python
students[:, 2].mean()
```

---

### Question 2

Find the maximum score.

```python
students[:, 2].max()
```

---

### Question 3

Find students who scored at least 80.

```python
students[students[:, 2] >= 80]
```

---

### Question 4

How many students scored at least 80?

```python
np.sum(students[:, 2] >= 80)
```

---

### Question 5

Find students who studied more than 5 hours.

```python
students[students[:, 1] > 5]
```

---

### Question 6

Find the average study hours.

```python
students[:, 1].mean()
```

---

# 46. The Most Important Connection So Far

Look at this:

```python
mean = X.mean(axis=0)
std = X.std(axis=0)

X_scaled = (X - mean) / std
```

It might initially look like some scary ML code.

But now break it down:

### Step 1

```python
X.mean(axis=0)
```

> Find the mean of every feature.

### Step 2

```python
X.std(axis=0)
```

> Find the standard deviation of every feature.

### Step 3

```python
X - mean
```

> Subtract each feature's mean.

Broadcasting handles the dimensions.

### Step 4

```python
/ std
```

> Divide each feature by its standard deviation.

Therefore:

```text
NumPy
  ↓
array operations
  ↓
statistics
  ↓
broadcasting
  ↓
mathematical formula
  ↓
ML preprocessing
```

That's exactly the mental bridge we're trying to build.

---

# 47. What You Should Be Comfortable With Now

At this stage, you should understand:

### Arithmetic

```python
X + 5
X - 5
X * 5
X / 5
X ** 2
```

### Comparisons

```python
X > 5
X >= 5
X < 5
X == 5
```

### Boolean filtering

```python
X[X > 5]
```

### Multiple conditions

```python
X[(X > 5) & (X < 10)]
```

### Statistics

```python
X.mean()
X.sum()
X.min()
X.max()
X.std()
X.var()
```

### Axis

```python
X.mean(axis=0)
X.mean(axis=1)
```

### Logical operations

```python
np.any()
np.all()
```

### Mathematical functions

```python
np.sqrt()
np.log()
np.exp()
np.abs()
np.clip()
```

### Random numbers

```python
np.random.default_rng()
```

### Broadcasting

You should understand **why**:

```python
X - mean
```

can work even though `X` and `mean` have different shapes.

---

# 48. The Mental Model I Want You to Build

Don't memorize:

> "`axis=0` means columns."

Instead think:

```text
I have:

X.shape = (1000, 20)

1000 → samples
20   → features
```

Then:

```python
X.mean(axis=0)
```

means:

> "I want one mean for each of my 20 features."

Therefore:

```text
input:

(1000, 20)

        ↓ mean(axis=0)

output:

(20,)
```

That's **real NumPy thinking**.

Similarly:

```python
X.mean(axis=1)
```

means:

> "I want one result for each of my 1000 samples."

Therefore:

```text
(1000, 20)

      ↓ mean(axis=1)

(1000,)
```

Once you start thinking in **shapes + axes + mathematical operations**, NumPy becomes much easier.

---

# Next: NumPy Part 4

We've now built the basic computational foundation.

The next lesson is going to be particularly important for your ML journey:

```text
                 NUMPY
                   │
          ┌────────┴─────────┐
          ↓                  ↓
      Reshaping          Combining
          │                  │
    reshape()           concatenate()
    reshape(-1, 1)      vstack()
    flatten()           hstack()
    ravel()             stack()
          │                  │
          └────────┬─────────┘
                   ↓
             DATASET MANIPULATION
                   ↓
             ML DATA PIPELINES
```

We'll also go deeply into the **difference between `(5,)`, `(5,1)`, and `(1,5)`**, because that seemingly tiny distinction causes a *lot* of confusion when beginners start using NumPy with scikit-learn and neural networks.
