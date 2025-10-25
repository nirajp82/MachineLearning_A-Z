# 🧮 Feature Scaling in Machine Learning: Normalization vs Standardization

## 📘 Introduction to Feature Scaling

Feature scaling is a crucial **preprocessing step** in machine learning.
It ensures all features (columns) have consistent ranges and units.

> ⚠️ Feature scaling is **always applied to columns**, not rows.
> Each feature column is scaled independently.

---

## 🔧 Types of Feature Scaling

There are many scaling techniques, but the two most common are:

* **Normalization**
* **Standardization**

---

## 🌈 Normalization

Normalization adjusts values so that each feature lies between **0 and 1**.

**Formula:**
[
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
]

**How it works:**

* Subtract the minimum value in the column.
* Divide by the difference between the maximum and minimum.

**Result:**
All values are scaled between **0 and 1**.

---

## 📏 Standardization

Standardization rescales data using the **mean** and **standard deviation**.

**Formula:**
[
x' = \frac{x - \text{mean}(x)}{\text{std}(x)}
]

**How it works:**

* Subtract the column’s mean.
* Divide by its standard deviation.

**Result:**
Most values fall between **-3 and 3**, though outliers may go beyond.

> In practice, **standardization** is often preferred.
> For simple intuition and visual examples, **normalization** is easier to demonstrate.

---

## 👥 Example: Comparing Individuals

We have a dataset with two columns — **Annual Income** and **Age** — for three people:

| Person | Annual Income ($) | Age (years) |
| :----: | :---------------: | :---------: |
|  Blue  |       70,000      |      45     |
| Purple |       60,000      |      44     |
|   Red  |       52,000      |      40     |

We want to find who the **Purple person** is most similar to — this is relevant for clustering algorithms.

---

### 🔍 Without Feature Scaling

| Comparison     | Salary Difference | Age Difference |
| :------------- | :---------------: | :------------: |
| Purple vs Blue |       10,000      |        1       |
| Purple vs Red  |       8,000       |        4       |

Since salary differences are much larger, it dominates the comparison —
even though age differences may be equally important.

This could wrongly suggest that **Purple is closer to Red**, just because 8,000 < 10,000.

---

## 🚨 Why Feature Scaling Matters

Comparing salary ($) to age (years) is like comparing **apples to oranges**.
Without scaling, features with large numeric ranges can overpower smaller ones,
leading to biased or misleading results.

---

## ⚙️ Applying Normalization

**Formula Reminder:**
[
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
]

### Normalized Annual Income

| Person | Normalized Income |
| :----: | :---------------: |
|  Blue  |        1.0        |
| Purple |       0.444       |
|   Red  |        0.0        |

### Normalized Age

| Person | Normalized Age |
| :----: | :------------: |
|  Blue  |       1.0      |
| Purple |      0.75      |
|   Red  |       0.0      |

---

### 🎯 After Normalization

* **Purple’s income (0.444)** is between Red (0.0) and Blue (1.0).
* **Purple’s age (0.75)** is closer to Blue (1.0) than to Red (0.0).

Now the comparison is **balanced** — both features contribute equally.

---

## ✅ Conclusion

Feature scaling ensures that no single feature dominates because of its range or units.
It leads to **fairer comparisons** and **better model performance**.

---

## 🗝️ Key Takeaways

| Concept             | Description                                                                 |
| :------------------ | :-------------------------------------------------------------------------- |
| **Applied to**      | Columns (features), never rows                                              |
| **Normalization**   | Scales data between 0 and 1                                                 |
| **Standardization** | Centers data (mean=0) and scales by standard deviation                      |
| **Typical range**   | Normalization: 0–1, Standardization: ~-3 to 3                               |
| **Purpose**         | Ensures fair comparison between features with different units or magnitudes |
