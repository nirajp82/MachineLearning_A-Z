# 🧠 Handling Missing Data in Machine Learning (Using Scikit-Learn)

This README summarizes and expands on **two Udemy lecture sessions** about handling missing data using **Scikit-Learn’s `SimpleImputer`**. It’s designed as a **revision and learning reference** for later review.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Why Handle Missing Data](#why-handle-missing-data)
3. [Common Strategies](#common-strategies)
4. [Using Scikit-Learn’s SimpleImputer](#using-scikit-learns-simpleimputer)

   * [Importing and Creating the Imputer](#importing-and-creating-the-imputer)
   * [Choosing a Strategy](#choosing-a-strategy)
5. [Applying the Imputer](#applying-the-imputer)

   * [Step 1: Fitting the Imputer](#step-1-fitting-the-imputer)
   * [Step 2: Transforming the Data](#step-2-transforming-the-data)
   * [Step 3: Updating the Dataset](#step-3-updating-the-dataset)
6. [General Rules and Best Practices](#general-rules-and-best-practices)
7. [Example Code](#example-code)

   * [Sample Output](#sample-output)
8. [Key Takeaways](#key-takeaways)

---

## 🧩 Introduction

In machine learning, **missing data** is a common issue that can disrupt model training and lead to poor predictions.
For example, consider the following dataset (`Data.csv`):

| Country | Age | Salary  | Purchased |
| ------- | --- | ------- | --------- |
| France  | 44  | 72000   | Yes       |
| Spain   | 27  | 48000   | No        |
| Germany | 30  | **NaN** | Yes       |
| Spain   | 38  | 61000   | No        |
| Germany | 40  | 63777   | Yes       |

Notice that the **Salary** for one customer is missing (`NaN`).

---

## ⚠️ Why Handle Missing Data

If missing data isn’t handled:

* The **machine learning model may fail to train** or throw errors.
* The **accuracy** of predictions could drop significantly.
* Many algorithms in Scikit-Learn cannot process missing (`NaN`) values directly.

---

## 🔧 Common Strategies

| Strategy                          | Description                                        | When to Use                                             |
| --------------------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| **Remove missing rows**           | Delete rows with missing values.                   | Works well when missing data is < 5% of dataset.        |
| **Replace with mean/median/mode** | Fill missing values with statistical replacements. | Recommended when data loss from removal is significant. |
| **Use predictive models**         | Predict missing values using other features.       | Used in advanced pipelines.                             |

In this lesson, we use the **mean strategy** — the most common approach for numerical data.

---

## 🧮 Using Scikit-Learn’s SimpleImputer

Scikit-Learn provides the **`SimpleImputer`** class for handling missing values efficiently.
It automates the process of replacing missing values with a chosen statistic (mean, median, or most frequent).

### 🧱 Importing and Creating the Imputer

```python
from sklearn.impute import SimpleImputer
import numpy as np
```

Create an imputer instance:

```python
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
```

**Parameters:**

* `missing_values=np.nan` → specifies what counts as missing.
* `strategy='mean'` → replaces missing values with the mean of that column.

---

### ⚙️ Choosing a Strategy

| Strategy Value    | Replacement Type           | Suitable For                          |
| ----------------- | -------------------------- | ------------------------------------- |
| `'mean'`          | Average of the column      | Numerical data                        |
| `'median'`        | Middle value of the column | Numerical data with outliers          |
| `'most_frequent'` | Most common value          | Categorical or ordinal data           |
| `'constant'`      | Fixed custom value         | When domain knowledge defines default |

---

## 🧰 Applying the Imputer

Once the imputer is defined, we must **fit it** to the dataset and **transform** the data to replace missing values.

### Step 1: Fitting the Imputer

The **`fit()`** method computes the replacement statistic (e.g., mean) for each selected column.

```python
imputer.fit(X[:, 1:3])  # Select only numerical columns (e.g., Age, Salary)
```

✅ `fit()` calculates the mean for each selected column.
🚫 It does **not** modify data — it just learns from it.

---

### Step 2: Transforming the Data

The **`transform()`** method replaces missing values with the computed statistics.

```python
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

✅ After `transform()`, all `NaN` values are replaced.
The method returns a **new NumPy array**, so you must reassign it to the same columns.

---

### Step 3: Updating the Dataset

After transformation:

* The original `X` feature matrix now has **no missing numerical values**.
* All categorical columns remain **unchanged**.

Example:

```python
print(X)
```

You’ll see that the previously missing salary has been replaced with the column’s mean value.

---

## 🧠 General Rules and Best Practices

1. **Always handle missing data before model training.**
2. **Only apply imputation to numerical columns** (strings or categories can cause errors).
3. **For large datasets**, use imputation to avoid losing valuable data.
4. **Avoid imputation on target variables (y)** — handle only feature matrix (X).
5. **Inspect results** after imputation to confirm correctness.

---

## 💻 Example Code

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Step 1: Load dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Feature matrix
y = dataset.iloc[:, -1].values   # Target variable

# Step 2: Create imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Step 3: Fit to numerical columns (Age, Salary)
imputer.fit(X[:, 1:3])

# Step 4: Transform and replace missing values
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Step 5: Verify
print(X)
```

---

### 🧾 Sample Output

Before Imputation:

```
[['France' 44.0 72000.0 'Yes']
 ['Spain' 27.0 48000.0 'No']
 ['Germany' 30.0 nan 'Yes']
 ['Spain' 38.0 61000.0 'No']
 ['Germany' 40.0 63777.0 'Yes']]
```

After Imputation:

```
[['France' 44.0 72000.0 'Yes']
 ['Spain' 27.0 48000.0 'No']
 ['Germany' 30.0 61194.25 'Yes']
 ['Spain' 38.0 61000.0 'No']
 ['Germany' 40.0 63777.0 'Yes']]
```

Here, the missing **salary** (`NaN`) has been replaced by the **mean salary**
→ `(72000 + 48000 + 61000 + 63777) / 4 = 61194.25`

---

## 🏁 Key Takeaways

| Concept                  | Description                                         |
| ------------------------ | --------------------------------------------------- |
| **Missing Data Problem** | Can cause model errors and poor performance.        |
| **SimpleImputer**        | Tool to replace missing values automatically.       |
| **fit()**                | Learns the statistic (e.g., mean) from data.        |
| **transform()**          | Applies imputation to replace missing values.       |
| **Best Practice**        | Apply only to numerical columns and verify results. |

---

