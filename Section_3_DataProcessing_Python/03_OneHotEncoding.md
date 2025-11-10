# One-Hot Encoding: Converting Categorical Data into Numbers

## Introduction
Machine learning models can only understand numbers — not words like “France” or “Spain.”  
So, before we train a model, we must **convert text data into numbers** in a way that makes sense and doesn’t confuse the model.

### Example Dataset
| Country | Age  | Salary  | Purchased |
|----------|------|----------|------------|
| France   | 44.0 | 72000.0  | No         |
| Spain    | 27.0 | 48000.0  | Yes        |
| Germany  | 30.0 | 54000.0  | No         |
| Spain    | 38.0 | 61000.0  | No         |
| Germany  | 40.0 | NaN      | Yes        |
| France   | 35.0 | 58000.0  | Yes        |
| Spain    | NaN  | 52000.0  | No         |
| France   | 48.0 | 79000.0  | Yes        |
| Germany  | 50.0 | 83000.0  | No         |
| France   | 37.0 | 67000.0  | Yes        |

The column **“Country”** contains text values.  
We need to turn these into numbers so the algorithm can use them.

---

## Why Simple Numbering Doesn’t Work
We could assign numbers like this:

```

France → 0
Spain → 1
Germany → 2

```

But this gives a **wrong idea to the model** — it might think Germany (2) is greater than Spain (1) or France (0), as if there’s an order or ranking.  
In reality, these are just **names**, not values with hierarchy.

---

## One-Hot Encoding: The Right Way
**One-Hot Encoding** fixes this by creating separate columns for each country and marking them with `1` or `0` depending on which country applies to that row.

### Encoded Data
| France | Spain | Germany | Age  | Salary  | Purchased |
|---------|-------|----------|-------|----------|------------|
| 1 | 0 | 0 | 44.0 | 72000.0 | No |
| 0 | 1 | 0 | 27.0 | 48000.0 | Yes |
| 0 | 0 | 1 | 30.0 | 54000.0 | No |
| 0 | 1 | 0 | 38.0 | 61000.0 | No |
| 0 | 0 | 1 | 40.0 | NaN | Yes |
| 1 | 0 | 0 | 35.0 | 58000.0 | Yes |
| 0 | 1 | 0 | NaN | 52000.0 | No |
| 1 | 0 | 0 | 48.0 | 79000.0 | Yes |
| 0 | 0 | 1 | 50.0 | 83000.0 | No |
| 1 | 0 | 0 | 37.0 | 67000.0 | Yes |

### Encoding Representation
```

France  = [1, 0, 0]
Spain   = [0, 1, 0]
Germany = [0, 0, 1]

```

This removes any implied ranking — all countries are treated equally and independently.

---

## ✅ Advantages
- Prevents false numerical relationships  
- Keeps data purely categorical  
- Improves interpretability for most algorithms  

## ⚠️ Drawback
If a categorical column has **many unique values** (e.g., hundreds of cities), one-hot encoding can create a **large number of columns** — this is known as the **curse of dimensionality**.

**Alternatives:**
- Target Encoding  
- Frequency Encoding  
- Embeddings (for deep learning)

---

## Encoding the “Purchased” Column
The **Purchased** column is the **target (output)** variable with only two possible values — “Yes” and “No.”

```

Yes → 1
No  → 0

````

This numeric conversion is **safe for binary classification** because it doesn’t introduce any false ordinal relationships.  
In other words, this numeric conversion does not create an ordering problem and works perfectly for binary outputs.

---

## Using Scikit-learn to Encode Categorical Data

### Tools
- **ColumnTransformer** — allows applying transformations to specific columns only, while keeping others unchanged. This is especially useful for datasets with both numeric and categorical data.  
- **OneHotEncoder** — converts categorical columns into one-hot encoded form, creating separate binary columns for each unique category. Best for features with no natural order (like Country).  
- **OrdinalEncoder** — converts categorical columns with a clear order or binary categories (like Purchased) into numeric values (No → 0, Yes → 1).  
- **LabelEncoder via FunctionTransformer** — used when you want to integrate LabelEncoder inside a ColumnTransformer. LabelEncoder normally works only on 1D arrays, so it needs wrapping to be compatible.

---

## Example Code: One-Hot and Ordinal Encoding
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# from sklearn.preprocessing import FunctionTransformer, LabelEncoder  # Uncomment if using LabelEncoder workaround

# ------------------------------------------
# 1️⃣ Create a Sample Dataset
# ------------------------------------------
data = {
    'Country': ['France', 'Spain', 'Germany', 'Spain', 'Germany', 'France', 'Spain', 'France', 'Germany', 'France'],
    'Age': [44.0, 27.0, 30.0, 38.0, 40.0, 35.0, np.nan, 48.0, 50.0, 37.0],
    'Salary': [72000.0, 48000.0, 54000.0, 61000.0, np.nan, 58000.0, 52000.0, 79000.0, 83000.0, 67000.0],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# ------------------------------------------
# 2️⃣ Handle Missing Numerical Data
# ------------------------------------------
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# ------------------------------------------
# 3️⃣ Encode Categorical Data
# ------------------------------------------
ct = ColumnTransformer(
    transformers=[
        ('country_encoder', OneHotEncoder(), ['Country']),       # One-hot encoding for Country
        ('purchase_encoder', OrdinalEncoder(), ['Purchased'])    # Ordinal encoding for Purchased
    ],
    remainder='passthrough'  # Keep Age and Salary as is
)

encoded_array = ct.fit_transform(df)
np.set_printoptions(suppress=True, precision=1)
````

---

## Optional: Using LabelEncoder inside ColumnTransformer

```python
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

def label_encode_column(col):
    le = LabelEncoder()
    return le.fit_transform(col).reshape(-1, 1)  # reshape to 2D for ColumnTransformer

ct_label = ColumnTransformer(
    transformers=[
        ('country_encoder', OneHotEncoder(), ['Country']),
        ('purchased_encoder', FunctionTransformer(label_encode_column), ['Purchased'])
    ],
    remainder='passthrough'
)

encoded_label = np.array(ct_label.fit_transform(df))
print(encoded_label)
```

---

## Explanation

### One-Hot Encoding for Country

```python
('country_encoder', OneHotEncoder(), ['Country'])
```

* Converts **Country** into multiple binary columns — one for each unique country.
* Example: France → [1,0,0], Spain → [0,1,0], Germany → [0,0,1].

### Ordinal Encoding for Purchased

```python
('purchase_encoder', OrdinalEncoder(), ['Purchased'])
```

* Converts `No → 0` and `Yes → 1`.
* Works for binary or ordered categorical columns.

### Using LabelEncoder with FunctionTransformer (Commented)

* **Why FunctionTransformer?**
  LabelEncoder expects 1D input, but ColumnTransformer provides a 2D array.
  Wrapping LabelEncoder in a function and using FunctionTransformer allows applying it inside ColumnTransformer.
* Output is reshaped to 2D to maintain compatibility.

### remainder='passthrough'

* Keeps numeric columns (**Age** and **Salary**) unchanged.

---

## Resulting Array

```
[[ 1.   0.   0.   0.  44. 72000. ]
 [ 0.   0.   1.   1.  27. 48000. ]
 [ 0.   1.   0.   0.  30. 54000. ]
 [ 0.   0.   1.   0.  38. 61000. ]
 [ 0.   1.   0.   1.  40. 63777.8]
 [ 1.   0.   0.   1.  35. 58000. ]
 [ 0.   0.   1.   0.  38.8 52000. ]
 [ 1.   0.   0.   1.  48. 79000. ]
 [ 0.   1.   0.   0.  50. 83000. ]
 [ 1.   0.   0.   1.  37. 67000. ]]
```

| Columns | Description                         |
| ------- | ----------------------------------- |
| 0–2     | One-hot encoded Country             |
| 3       | Numeric Purchased (0 = No, 1 = Yes) |
| 4–5     | Age and Salary                      |

---

## Key Takeaways

* Machine learning models can’t handle text, so categorical data must be converted into numbers.
* Simple numbering (0, 1, 2) is misleading because it creates false order relationships.
* **One-Hot Encoding** is the right choice — it converts text values into binary columns.
* For binary outputs (like “Yes/No”), you can safely use 1 and 0.
* Use **ColumnTransformer** and **OneHotEncoder** from Scikit-learn to do this efficiently.
* The **ColumnTransformer** class is used to apply transformations to specific columns in a dataset.
* The `'transformers'` argument specifies the type of transformation, the transformer class, and the column indexes to transform.
* The `'remainder'` argument with value `'passthrough'` keeps columns that are not transformed.
* **OrdinalEncoder** works for binary or ordered categorical columns.
* **LabelEncoder with FunctionTransformer** allows integrating LabelEncoder into ColumnTransformer pipelines.
* Remember: one-hot encoding increases the number of columns — for many unique categories, this can make your dataset large.
* The **fit_transform()** method of ColumnTransformer allows us to fit and transform at once.
* The output of `fit_transform()` is not a NumPy array by default; since most ML models expect NumPy arrays, we convert it using `np.array()`.

---
