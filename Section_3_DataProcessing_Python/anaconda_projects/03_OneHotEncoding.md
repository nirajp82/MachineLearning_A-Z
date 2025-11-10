```markdown
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

---

## Using Scikit-learn to Encode Categorical Data

### Tools
- **ColumnTransformer** — apply transformations to specific columns only while keeping others unchanged.  
- **OneHotEncoder** — converts categorical features (like Country) into binary columns.  
- **OrdinalEncoder** — converts ordered/binary categorical columns (like Purchased) into numeric values.  
- **LabelEncoder + FunctionTransformer** — used to integrate LabelEncoder inside ColumnTransformer when needed.

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

### Optional: Using LabelEncoder inside ColumnTransformer

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

## Explanation of Components

### One-Hot Encoding for Country

```python
('country_encoder', OneHotEncoder(), ['Country'])
```

* Converts Country into multiple binary columns — one for each unique value.
* Example: France → [1,0,0], Spain → [0,1,0], Germany → [0,0,1].

### Ordinal Encoding for Purchased

```python
('purchase_encoder', OrdinalEncoder(), ['Purchased'])
```

* Converts `No → 0`, `Yes → 1`.
* Suitable for binary or ordered categorical columns.

### LabelEncoder with FunctionTransformer

* **Problem:** LabelEncoder expects 1D input, but ColumnTransformer provides 2D.
* **Solution:** Wrap LabelEncoder inside a function and use **FunctionTransformer** to make it compatible.
* Output reshaped to 2D for proper integration.

### remainder='passthrough'

* Keeps numeric columns (Age and Salary) unchanged.

---

## Resulting Encoded Array

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

* ML models can’t handle text → categorical data must be numeric.
* Simple numbering (0, 1, 2) introduces false relationships.
* **One-Hot Encoding** is ideal for unordered categorical features.
* Binary columns (e.g., “Yes/No”) can safely be encoded as `1` and `0`.
* Use **ColumnTransformer** to apply different transformations per column.
* `remainder='passthrough'` keeps untransformed columns.
* **OrdinalEncoder** works for binary/ordered categories.
* **LabelEncoder + FunctionTransformer** allows LabelEncoder in pipelines.
* Beware: one-hot encoding increases column count — can cause dimensionality issues.
* Use `fit_transform()` to fit and transform data in one step.
* The result should be converted to a NumPy array for compatibility with ML models.

---

```
```
