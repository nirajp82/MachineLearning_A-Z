# Dataset Splitting and Feature Scaling

### Introduction
Getting your data ready is a key first step in machine learning. Two important parts of this are splitting the data into training and test sets, and scaling the features. Doing these steps properly helps your model perform better and avoids mistakes when testing on new data.

### Splitting the Dataset

Dataset splitting is the process of dividing your data into **two separate groups** so you can train and evaluate your model properly.

* **Training set:** Used to teach the model patterns and relationships in the data.
* **Test set:** Used to check how well the model performs on **new, unseen examples**.

Think of it like studying for an exam: the **training set** is your study material, and the **test set** is the actual exam to see if you really understand the concepts.

> Splitting ensures the model learns from one set of data but is evaluated on completely unseen data, giving a fair measure of its performance.

**Example Table:**
Suppose you have 10 rows of data on customer purchases:

| Row | Age | Income | Bought Product |
| --- | --- | ------ | -------------- |
| 1   | 24  | 40,000 | Yes            |
| 2   | 30  | 48,000 | No             |
| 3   | 22  | 39,000 | Yes            |
| 4   | 27  | 52,000 | No             |
| 5   | 35  | 80,000 | Yes            |
| 6   | 29  | 61,000 | No             |
| 7   | 41  | 54,000 | Yes            |
| 8   | 33  | 58,000 | No             |
| 9   | 26  | 42,000 | Yes            |
| 10  | 28  | 50,000 | No             |

A common split is:

* **Rows 1–8** → Training set
* **Rows 9–10** → Test set

Now, the model **learns only from the training set**. The test set is kept separate to **evaluate real performance**.

**Tip:** If your dataset is small, you can use **cross-validation** to make the most of your data while still testing generalization.

---

### Feature Scaling

Feature scaling is the process of adjusting the range of numerical features so that they are on a similar scale. Many machine learning algorithms, such as K-Nearest Neighbors (KNN), Logistic Regression, and Neural Networks, are sensitive to the magnitude of numbers.

Without scaling, features with larger values can dominate the learning process, while smaller-valued features may be ignored, even if they are important.

- Feature: In machine learning, a feature is any individual measurable property or characteristic of your data. For example: Age, Income, Height, Weight, or Exam Score. Each column in your dataset usually represents one feature.
- Scaling: Scaling means changing the range of values without changing the relationships between them. You are essentially compressing or stretching the numbers so they are comparable.

**Example:**

* Age: 20–60
* Income: 30,000–120,000

If we feed this data directly into a model without scaling, the model may focus mostly on income because the numbers are larger, ignoring age.
- The Income feature has much larger numbers than Age.
- The model may consider Income more “important” simply because of its magnitude, not because it actually matters more.
- his can bias the predictions and reduce model accuracy.

Scaling fixes this by bringing all features to a comparable scale, without changing the relationships between values. After scaling, the model can learn equally from Age and Income.

---

### Popular Scaling Methods

#### 1️⃣ Standardization (Z-score)

Standardization centers data around **0** and scales it based on the **standard deviation**.

**Formula:**

```
X_scaled = (X - μ) / σ
```

Where:

* `X` = original value
* `μ` = mean of the training data
* `σ` = standard deviation of the training data
  
**Examples:**

* Income: mean = 60,000, std = 15,000

  * Original income = 75,000

  ```
  (75,000 - 60,000) / 15,000 = 1
  ```

* Age: mean = 35, std = 10

  * Original age = 50

  ```
  (50 - 35) / 10 = 1.5
  ```

* Exam score: mean = 70, std = 10

  * Original score = 80

  ```
  (80 - 70) / 10 = 1
  ```

**Interpretation:** A Z-score of **1** means the value is **1 standard deviation above the mean**.

---

#### 2️⃣ Min-Max Scaling

Min-Max scaling rescales values to a **fixed range**, usually 0–1.

**Formula:**

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Examples:**

* Age: min = 20, max = 60

  * Original age = 40

  ```
  (40 - 20) / (60 - 20) = 0.5
  ```

* Income: min = 30,000, max = 120,000

  * Original income = 75,000

  ```
  (75,000 - 30,000) / (120,000 - 30,000) = 0.5
  ```

* Exam score: min = 50, max = 100

  * Original score = 80

  ```
  (80 - 50) / (100 - 50) = 0.6
  ```

**Tip:** Standardization is better if your data has **outliers**, while Min-Max scaling is more sensitive to them.

---

### Why Scale After Splitting

**Always scale after splitting.**

**Why:** Scaling before splitting uses information from the test set, which **leaks data** into training. This makes your model appear more accurate than it really is.

**Correct Steps:**

1. Split data into training and test sets.
2. Fit the scaler **only on training data**.
3. Transform both training and test sets using the scaler.

**Wrong:** Scaling first, then splitting — this allows the test set to influence scaling and gives an unfair advantage.

---

### Key Takeaways

* Always **split first, then scale**.
* Fit scalers **on training data only**, then transform both sets.
* Scaling puts all features on a level playing field, helping the model learn efficiently.
* The test set must remain separate to measure **real-world performance**.
* Use cross-validation for small datasets to maximize data usage.

---

### Example Code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example data: columns are Age, Income
data = np.array([
    [24, 40000],
    [30, 48000],
    [22, 39000],
    [27, 52000],
    [35, 80000],
    [29, 61000],
    [41, 54000],
    [33, 58000],
    [26, 42000],
    [28, 50000]
])

# 1. Split the dataset (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 2. Create a scaler (StandardScaler or MinMaxScaler)
scaler = StandardScaler()  # or MinMaxScaler()

# 3. Fit only on training data
scaler.fit(train_data)

# 4. Transform both training and test sets
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# Print results
print("Original training data:\n", train_data)
print("\nScaled training data:\n", train_scaled)
print("\nOriginal test data:\n", test_data)
print("\nScaled test data:\n", test_scaled)
```
