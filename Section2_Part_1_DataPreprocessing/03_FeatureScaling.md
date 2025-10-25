````markdown
# 🧮 Introduction to Feature Scaling

Feature scaling is a crucial preprocessing step in machine learning. It is always applied to columns of data, never across rows. This means that each feature column is scaled independently to ensure consistent ranges and units across features.

> ⚠️ **Note:** Feature scaling is always applied to columns.  

Feature scaling is always applied to columns. For example, scaling would be applied to this column, to this column, and so forth, but never across the data inside a row. Remember, feature scaling is always applied to columns.

---

## 🔧 Types of Feature Scaling

There are multiple techniques for feature scaling, but the two main ones are **normalization** and **standardization**.

### 🌟 Normalization

Normalization involves taking the minimum value in a column, subtracting it from every value in that column, and then dividing by the difference between the maximum and minimum values.  
This process adjusts every value in the column so that the resulting values lie between **0 and 1**.

**Formula:**
```text
Normalized value = (value - min) / (max - min)
````

> 💡 **Tip:** Normalization is often demonstrated in intuition tutorials.

### 🌟 Standardization

Standardization is a similar process, but instead of subtracting the minimum, it subtracts the **average (mean)** of the column and divides by the **standard deviation**.
As a result, most values in the column will lie between **-3 and 3**. However, extreme values or outliers can fall outside this range.

**Formula:**

```text
Standardized value = (value - mean) / std
```

> 💡 **Tip:** Standardization is often used in practical tutorials.

---

## 👥 Illustrative Example: Comparing Individuals

Consider a dataset with two columns: **annual income** and **age**. We have three individuals: a **blue person**, a **purple person**, and a **red person**. Their data is as follows:

* **Blue person:** $70,000 annual income, 45 years old
* **Purple person:** $60,000 annual income, 44 years old
* **Red person:** $52,000 annual income, 40 years old

The task is to determine which individual the purple person is most similar to based on this data. This is relevant for clustering tasks and algorithms.

---

## 🔍 Comparing Differences Without Scaling

Let's examine the differences between the purple person and the others:

* **Salary difference with blue person:** $10,000
* **Salary difference with red person:** $8,000
* **Age difference with blue person:** 1 year
* **Age difference with red person:** 4 years

Because salary differences are much larger in magnitude than age differences, unscaled features can cause the salary to dominate the similarity measure.

Due to the large salary differences compared to age differences, one might erroneously conclude that the purple person is closer to the red person because
`8,000 < 10,000`. This ignores the smaller but potentially important age differences.

> ⚠️ **Note:** Always consider both features — large differences can bias similarity measures.

---

## 🚨 Importance of Feature Scaling

To avoid such misleading conclusions, feature scaling is necessary. Comparing salaries directly to years is like comparing **apples to oranges**. Even if units are consistent, features may relate to different concepts and scales, making scaling essential.

---

## ⚙️ Applying Normalization

Recall the normalization formula:

```text
Normalized value = (value - min) / (max - min)
```

We apply this formula to each column independently.

After normalizing the annual income column, the values become:

* **Blue person:** 1.0
* **Purple person:** 0.444
* **Red person:** 0.0

After normalizing the age column, the values become:

* **Blue person:** 1.0
* **Purple person:** 0.75
* **Red person:** 0.0

---

## 🎯 Comparing Individuals After Normalization

Now, the purple person is almost exactly in the middle between the red and blue persons in terms of income (**0.444**).
In terms of age, the purple person is closer to the blue person (**0.75 vs. 1.0 and 0.0**).
This balanced scaling allows for fairer comparison across features.

---

## ✅ Conclusion

This simple example illustrates the importance of feature scaling in machine learning. Proper scaling ensures that features with different units or magnitudes contribute appropriately to similarity measures and model performance.

---

## 🗝️ Key Takeaways

* ⚡ Feature scaling is always applied to **columns**, never across rows.
* 🔹 **Normalization** rescales data to a 0 to 1 range by subtracting the minimum and dividing by the range.
* 🔹 **Standardization** rescales data by subtracting the mean and dividing by the standard deviation, typically resulting in values between -3 and 3.
* ⚡ Feature scaling is essential to ensure fair comparison between features with different units or magnitudes.
