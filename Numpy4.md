# NumPy — Part 4: Reshaping, Flattening & Combining Arrays

This lesson is **very important for ML**.

A lot of beginner ML errors aren't because the mathematics is difficult—they happen because the data has the **wrong shape**.

For example:

```text
(5,)
(5, 1)
(1, 5)
```

These look almost identical to a beginner.

To NumPy—and especially to ML libraries—they can mean **three different things**.

Let's make this crystal clear.

---

# 1. First: What Does Shape Actually Mean?

Suppose:

```python
import numpy as np

x = np.array([10, 20, 30, 40, 50])
```

Check:

```python
x.shape
```

Output:

```text
(5,)
```

This means:

> There is one dimension containing 5 elements.

Visualize it as:

```text
[10 20 30 40 50]
```

This is a **1D array**.

```python
x.ndim
```

→

```text
1
```

---

# 2. `(5,)` vs `(5,1)`

Now look at:

```python
x = np.array([
    [10],
    [20],
    [30],
    [40],
    [50]
])
```

Check:

```python
x.shape
```

Output:

```text
(5, 1)
```

This is now a **2D array**.

Visualize:

```text
10
20
30
40
50
```

It has:

```text
5 rows
1 column
```

So:

```text
(5,)   → 1D
(5,1)  → 2D column
```

This distinction is extremely important.

---

# 3. `(1,5)`

Now:

```python
x = np.array([
    [10, 20, 30, 40, 50]
])
```

Shape:

```python
x.shape
```

→

```text
(1, 5)
```

This means:

```text
1 row
5 columns
```

Visual:

```text
10 20 30 40 50
```

Therefore:

```text
(5,)    → 1D array
(5,1)   → column
(1,5)   → row
```

Memorize this.

---

# 4. Why Does ML Care?

Suppose you have 5 students and one feature:

```text
Study Hours

5
7
3
8
4
```

You might represent it as:

```python
X = np.array([5, 7, 3, 8, 4])
```

Shape:

```text
(5,)
```

But an ML dataset is generally represented as:

```text
(samples, features)
```

So if you have:

```text
5 samples
1 feature
```

the natural 2D representation is:

```text
(5, 1)
```

Like:

```python
X = np.array([
    [5],
    [7],
    [3],
    [8],
    [4]
])
```

This is one of the reasons reshaping matters.

---

# 5. `reshape()`

The most important tool for changing an array's shape is:

```python
reshape()
```

Start with:

```python
x = np.array([1, 2, 3, 4, 5, 6])
```

Shape:

```text
(6,)
```

We can reshape it:

```python
x.reshape(2, 3)
```

Result:

```text
[[1 2 3]
 [4 5 6]]
```

Shape:

```text
(2, 3)
```

---

# 6. What Happened?

Originally:

```text
[1 2 3 4 5 6]
```

NumPy reorganized the same 6 numbers:

```text
1 2 3
4 5 6
```

Notice:

> **The data didn't change. The shape changed.**

That's an extremely important concept.

---

# 7. Another Reshape

```python
x.reshape(3, 2)
```

gives:

```text
[[1 2]
 [3 4]
 [5 6]]
```

Shape:

```text
(3, 2)
```

Same six numbers.

Different arrangement.

---

# 8. The Golden Rule of Reshape

You cannot magically create or destroy elements.

If:

```python
x.size
```

is:

```text
6
```

then the new shape must also contain:

```text
6 elements
```

Therefore these work:

```python
x.reshape(2, 3)
x.reshape(3, 2)
x.reshape(6, 1)
x.reshape(1, 6)
```

But:

```python
x.reshape(4, 2)
```

doesn't work.

Why?

```text
4 × 2 = 8
```

but you only have 6 elements.

---

# 9. The `-1` Trick

NumPy allows you to let it calculate one dimension.

Suppose:

```python
x = np.array([1, 2, 3, 4, 5, 6])
```

Instead of:

```python
x.reshape(6, 1)
```

you can write:

```python
x.reshape(-1, 1)
```

NumPy asks:

> "If there are 6 total elements and one column, how many rows should there be?"

Answer:

```text
6
```

So:

```text
(6,1)
```

---

# 10. Why `reshape(-1, 1)` Is So Common in ML

Suppose:

```python
study_hours = np.array([5, 7, 3, 8, 4])
```

Currently:

```text
(5,)
```

We want:

```text
(samples, features)
```

There are 5 samples and 1 feature.

So:

```python
study_hours = study_hours.reshape(-1, 1)
```

Now:

```python
study_hours.shape
```

→

```text
(5, 1)
```

This pattern is worth memorizing:

```python
X = X.reshape(-1, 1)
```

means:

> Turn this into a 2D column while letting NumPy determine the number of rows.

---

# 11. `reshape(1, -1)`

The opposite:

```python
x.reshape(1, -1)
```

means:

> Make one row and figure out the number of columns automatically.

For:

```python
x = np.array([10, 20, 30, 40, 50])
```

we get:

```text
[[10 20 30 40 50]]
```

Shape:

```text
(1,5)
```

So:

```text
reshape(-1,1) → column
reshape(1,-1) → row
```

---

# 12. Real ML Example

Imagine you're predicting salary from experience.

```python
experience = np.array([1, 2, 5, 8, 12])
```

This is:

```text
(5,)
```

We have:

```text
5 employees
1 feature
```

Convert it:

```python
X = experience.reshape(-1, 1)
```

Now:

```text
X.shape
```

→

```text
(5, 1)
```

Perfect ML feature matrix.

---

# 13. Multiple Features

Suppose we have:

```python
X = np.array([
    [22, 1],
    [25, 2],
    [29, 5],
    [35, 8],
    [42, 12]
])
```

Shape:

```text
(5,2)
```

Meaning:

```text
5 samples
2 features
```

We **don't need to reshape this** because it's already a 2D feature matrix.

---

# 14. Flattening

Now let's go in the opposite direction.

Suppose:

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

Shape:

```text
(2,3)
```

We want one long array:

```text
[1 2 3 4 5 6]
```

Use:

```python
X.flatten()
```

Result:

```text
[1 2 3 4 5 6]
```

Shape:

```text
(6,)
```

---

# 15. Why Flattening Is Important

Imagine an image.

Suppose an image is:

```text
28 × 28 pixels
```

Its shape is:

```text
(28, 28)
```

That's:

```text
784 pixels
```

A traditional ML model might require:

```text
(784,)
```

So:

```python
image.flatten()
```

converts:

```text
28 × 28
```

into:

```text
784
```

This is why you will hear the word **flatten** frequently in ML and neural networks.

---

# 16. `flatten()` vs `ravel()`

Both can turn multidimensional arrays into 1D arrays.

```python
X.flatten()
```

and:

```python
X.ravel()
```

For beginners:

> Think of both as "give me a 1D version."

There is, however, an important technical difference involving whether NumPy returns a copy or a view.

For now, remember:

```text
flatten → generally returns a copy
ravel   → may return a view when possible
```

We'll revisit **views vs copies** later because it is an important NumPy concept.

---

# 17. Combining Arrays

Now imagine you collect data in batches.

Batch 1:

```python
batch1 = np.array([
    [20, 40000],
    [25, 50000]
])
```

Batch 2:

```python
batch2 = np.array([
    [30, 70000],
    [35, 90000]
])
```

You want one dataset:

```text
20 40000
25 50000
30 70000
35 90000
```

You can use:

```python
np.concatenate([batch1, batch2])
```

Result:

```text
[[20 40000]
 [25 50000]
 [30 70000]
 [35 90000]]
```

---

# 18. `axis` Appears Again

For:

```python
batch1.shape
```

we have:

```text
(2,2)
```

and:

```python
batch2.shape
```

also:

```text
(2,2)
```

When we concatenate along:

```python
axis=0
```

we add more **rows**:

```python
np.concatenate([batch1, batch2], axis=0)
```

Result shape:

```text
(4,2)
```

This is exactly what you'd expect when adding more samples.

---

# 19. Concatenating Columns

Suppose:

```python
age = np.array([
    [20],
    [25],
    [30]
])

salary = np.array([
    [40000],
    [50000],
    [70000]
])
```

Both have:

```text
(3,1)
```

We can combine them horizontally:

```python
np.concatenate([age, salary], axis=1)
```

Result:

```text
[[20 40000]
 [25 50000]
 [30 70000]]
```

Shape:

```text
(3,2)
```

We have created:

```text
Age + Salary
```

as two features.

---

# 20. `vstack()`

NumPy provides convenient functions.

`vstack` means:

> Stack vertically.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.vstack([a, b])
```

Result:

```text
[[1 2 3]
 [4 5 6]]
```

Shape:

```text
(2,3)
```

Think:

```text
a
↓
b
```

---

# 21. `hstack()`

`hstack` means:

> Stack horizontally.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.hstack([a, b])
```

Result:

```text
[1 2 3 4 5 6]
```

For 1D arrays, this can sometimes behave differently from what beginners intuitively expect, so don't blindly assume "horizontal" always means "two columns."

For ML datasets, explicit 2D shapes are often clearer.

For example:

```python
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])

np.hstack([a, b])
```

gives:

```text
[[1 4]
 [2 5]
 [3 6]]
```

Shape:

```text
(3,2)
```

Now the behavior is visually obvious.

---

# 22. `stack()`

There's another function:

```python
np.stack()
```

This is slightly different.

Suppose:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
```

Then:

```python
np.stack([a, b])
```

gives:

```text
[[1 2 3]
 [4 5 6]]
```

Shape:

```text
(2,3)
```

But:

```python
np.stack([a, b], axis=1)
```

gives:

```text
[[1 4]
 [2 5]
 [3 6]]
```

Shape:

```text
(3,2)
```

The key difference is that `stack` creates a **new axis**.

---

# 23. Don't Memorize All Combining Functions Yet

For now, build this mental map:

```text
concatenate
     │
     ├── axis=0 → add rows
     └── axis=1 → add columns

vstack → stack vertically

hstack → stack horizontally

stack → create a new dimension
```

We'll use these repeatedly in practical examples.

---

# 24. Dataset Construction Example

Imagine we're collecting employee data.

Initially:

```python
employees1 = np.array([
    [22, 40000],
    [25, 50000]
])
```

Later:

```python
employees2 = np.array([
    [30, 70000],
    [35, 90000]
])
```

Combine:

```python
employees = np.concatenate(
    [employees1, employees2],
    axis=0
)
```

Now:

```text
[[22 40000]
 [25 50000]
 [30 70000]
 [35 90000]]
```

Shape:

```text
(4,2)
```

This is a realistic dataset-building operation.

---

# 25. Adding a New Feature

Suppose we already have:

```python
age_salary = np.array([
    [22, 40000],
    [25, 50000],
    [30, 70000],
    [35, 90000]
])
```

Now we obtain experience:

```python
experience = np.array([
    [1],
    [2],
    [5],
    [8]
])
```

Combine:

```python
X = np.concatenate(
    [age_salary, experience],
    axis=1
)
```

Now:

```text
[[22 40000     1]
 [25 50000     2]
 [30 70000     5]
 [35 90000     8]]
```

Shape:

```text
(4,3)
```

We've just added a feature to our ML dataset.

---

# 26. Splitting Arrays

The opposite of combining is splitting.

NumPy has:

```python
np.split()
```

For example:

```python
x = np.array([1, 2, 3, 4, 5, 6])
```

Split into two equal pieces:

```python
np.split(x, 2)
```

Result:

```text
[array([1, 2, 3]), array([4, 5, 6])]
```

---

# 27. `array_split()`

`np.split()` requires the split to work evenly in the relevant dimension.

`np.array_split()` is more flexible.

```python
x = np.array([1, 2, 3, 4, 5])
```

Then:

```python
np.array_split(x, 2)
```

can divide the data into chunks even when the sizes aren't equal.

This can be useful for dividing datasets into pieces.

---

# 28. Transpose

Another extremely important operation:

```python
.T
```

Suppose:

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

Shape:

```text
(2,3)
```

Transpose:

```python
X.T
```

Result:

```text
[[1 4]
 [2 5]
 [3 6]]
```

Shape:

```text
(3,2)
```

It essentially swaps:

```text
rows ↔ columns
```

---

# 29. Why Transpose Matters in ML

You may encounter:

```text
X.shape = (1000, 20)
```

meaning:

```text
1000 samples
20 features
```

After transpose:

```python
X.T.shape
```

becomes:

```text
(20,1000)
```

Now:

```text
20 features
1000 samples
```

Transpose is fundamental in linear algebra and therefore fundamental to ML.

---

# 30. A Shape Story

Let's create a simple mental picture.

Suppose:

```text
100 students
5 features
```

Then:

```text
X.shape = (100,5)
```

Think:

```text
                 FEATURES
       F1   F2   F3   F4   F5
       ↓    ↓    ↓    ↓    ↓
S1    [ ]  [ ]  [ ]  [ ]  [ ]
S2    [ ]  [ ]  [ ]  [ ]  [ ]
S3    [ ]  [ ]  [ ]  [ ]  [ ]
...
S100  [ ]  [ ]  [ ]  [ ]  [ ]
```

So:

```text
rows    → samples
columns → features
```

This is the fundamental ML dataset structure.

---

# 31. Then Reshaping Makes Sense

Suppose you only have one feature:

```text
100 students
1 feature
```

You want:

```text
(100,1)
```

So:

```python
feature = feature.reshape(-1, 1)
```

The `-1` says:

> "NumPy, you figure out the number of samples."

The `1` says:

> "There is exactly one feature."

That's a much better way to remember it than memorizing syntax.

---

# 32. A Very Common Beginner Error

Suppose:

```python
X = np.array([1, 2, 3, 4, 5])
```

Then:

```python
X.shape
```

is:

```text
(5,)
```

A beginner may think:

> "That's 5 rows."

Not quite.

It is simply:

> A one-dimensional array with 5 elements.

If you explicitly want 5 rows and 1 column:

```python
X.reshape(-1, 1)
```

giving:

```text
(5,1)
```

This distinction becomes especially important when passing data into ML algorithms.

---

# 33. Mini ML Pipeline

Let's put today's concepts together.

We start with raw feature data:

```python
study_hours = np.array([5, 7, 3, 8, 4])
```

Check:

```python
study_hours.shape
```

→

```text
(5,)
```

Convert into ML-style feature matrix:

```python
X = study_hours.reshape(-1, 1)
```

Now:

```text
(5,1)
```

Suppose we add another feature:

```python
sleep_hours = np.array([7, 6, 8, 6, 7]).reshape(-1, 1)
```

Now combine:

```python
X = np.concatenate(
    [X, sleep_hours],
    axis=1
)
```

Result:

```text
[[5 7]
 [7 6]
 [3 8]
 [8 6]
 [4 7]]
```

Shape:

```text
(5,2)
```

We now have:

```text
5 students
2 features
```

---

# 34. This Is How You Should Think

Not:

> "I need to remember `reshape(-1,1)`."

Instead:

```text
I have:
5 samples
1 feature

Therefore:
shape should be (5,1)

How do I achieve it?
reshape(-1,1)
```

That is **understanding rather than memorization**.

---

# 35. Practice Challenge

Try this yourself before looking at the answers.

```python
import numpy as np

age = np.array([20, 25, 30, 35, 40])

salary = np.array([
    40000,
    50000,
    70000,
    90000,
    120000
])

experience = np.array([1, 2, 5, 8, 12])
```

### Challenge 1

What are the shapes of all three arrays?

---

### Challenge 2

Convert each into a 2D column.

Hint:

```python
.reshape(?, ?)
```

---

### Challenge 3

Combine all three into:

```text
(samples, features)
```

so the final shape is:

```text
(5,3)
```

---

### Challenge 4

Transpose the resulting dataset.

What should its shape become?

---

### Challenge 5

Flatten the dataset.

How many values should it contain?

---

# 36. Solution

### Step 1

Initially:

```text
age.shape
→ (5,)

salary.shape
→ (5,)

experience.shape
→ (5,)
```

---

### Step 2

Convert:

```python
age = age.reshape(-1, 1)
salary = salary.reshape(-1, 1)
experience = experience.reshape(-1, 1)
```

Each becomes:

```text
(5,1)
```

---

### Step 3

Combine:

```python
X = np.concatenate(
    [age, salary, experience],
    axis=1
)
```

Result:

```text
[[20  40000      1]
 [25  50000      2]
 [30  70000      5]
 [35  90000      8]
 [40 120000     12]]
```

Shape:

```text
(5,3)
```

---

### Step 4

Transpose:

```python
X.T
```

Shape:

```text
(3,5)
```

---

### Step 5

Flatten:

```python
X.flatten()
```

Shape:

```text
(15,)
```

because:

```text
5 × 3 = 15
```

---

# 37. NumPy Knowledge Map So Far

You now have:

```text
                         NumPy
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
          Arrays                     Operations
             │                           │
        ┌────┼────┐              ┌───────┼───────┐
        ↓    ↓    ↓              ↓       ↓       ↓
      ndim shape size         arithmetic comparison statistics
        │    │    │              │       │       │
        └────┴────┘              │       │       │
             │                   │       │       │
             ↓                   ↓       ↓       ↓
         Indexing             masking  where    mean/std
         Slicing
             │
             ↓
        Reshaping
             │
      ┌──────┼──────┐
      ↓      ↓      ↓
   reshape flatten transpose
             │
             ↓
       Combining
             │
      ┌──────┼──────┐
      ↓      ↓      ↓
 concatenate stack  v/hstack
```

And you're now at the point where NumPy starts becoming genuinely useful for ML.

---

# 38. What Comes Next

The next lesson will be **NumPy Part 5: Linear Algebra for ML**.

This is where we'll connect NumPy to the mathematics underneath ML:

```text
Vectors
   ↓
Dot Product
   ↓
Matrix Multiplication
   ↓
Transpose
   ↓
Matrix-Vector multiplication
   ↓
Linear equations
   ↓
Linear Regression
   ↓
Neural-network intuition
```

We'll take something like:

```text
Prediction = Xw + b
```

and **build it ourselves using NumPy**, rather than treating it as a mysterious ML formula.

That will be the point where you'll start seeing how **NumPy → Linear Algebra → Machine Learning** fit together.
