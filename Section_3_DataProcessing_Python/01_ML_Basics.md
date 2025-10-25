# 🧮 Features and Dependent Variable in Machine Learning
---

## 1️⃣ Understanding Features and Dependent Variable

In any dataset used to train a machine learning model:

- **Features (Independent Variables):**  
  - Columns used to **predict the outcome**  
  - Provide the information that the model uses to learn  
  - Usually occupy the **first columns** of the dataset

- **Dependent Variable (Target / Outcome):**  
  - The value we want to **predict**  
  - Usually the **last column** in the dataset  

> ⚠️ **Note:** Features = independent variables, dependent variable = target.

---

### Example Dataset

| Country | Age | Salary | Purchased |
|:-------:|:---:|:------:|:---------:|
| France  | 44  | 72000  | Yes       |
| Spain   | 27  | 48000  | No        |
| Germany | 30  | 54000  | Yes       |
| Spain   | 38  | 61000  | No        |
| Germany | 40  | 65000  | Yes       |

- **Features:** `Country`, `Age`, `Salary`  
- **Dependent Variable:** `Purchased`  

---

## 2️⃣ Creating Feature Matrix and Dependent Variable Vector

We separate the dataset into two entities:

| Entity | Description | Representation |
|:------:|:-----------:|:--------------|
| **Feature Matrix** | Contains all features (independent variables) used for prediction | `X` |
| **Dependent Variable Vector** | Contains the target values we want to predict | `y` |

### Python Example

```python
# Importing dataset
import pandas as pd
dataset = pd.read_csv('data.csv')

# Creating feature matrix (X) and dependent variable vector (y)
X = dataset.iloc[:, :-1].values  # All columns except last
y = dataset.iloc[:, -1].values   # Only the last column
````

> ⚡ `X` will have columns: `Country`, `Age`, `Salary`
> ⚡ `y` will have the column: `Purchased`

---

### 🗝️ Key Takeaways

* Features are **inputs** to the model, dependent variable is the **output**
* Always separate the dataset into **X (features)** and **y (target)** before modeling
* Ensures clear distinction between what is **given** to the model and what is to be **predicted**

```
```
