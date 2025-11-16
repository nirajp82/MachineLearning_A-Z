# Dataset Splitting and Feature Scaling

### Introduction
Getting your data ready is a key first step in machine learning. Two important parts of this are splitting the data into training and test sets, and scaling the features. Doing these steps properly helps your model perform better and avoids mistakes when testing on new data.

### Splitting the Dataset

Splitting the dataset means dividing all your data into two groups:
- **Training set:** Used to teach your model how to find patterns.
- **Test set:** Used to test how well your model works on new examples it hasn’t seen before.

The idea is like studying for a test: you learn using your textbook (training set), but on exam day (test set), you face new questions to check your actual understanding.

**Example Table:**  
Say you have 10 rows of data on customer purchases:

| Row | Age | Income | Bought Product |
|-----|-----|--------|---------------|
| 1   | 24  | 40,000 | Yes           |
| ... | ... | ...    | ...           |
|10   | 41  | 54,000 | No            |

You could split this into:
- Rows 1–8 for training (learning)
- Rows 9–10 for testing (checking)

Now your model only learns from the first 8 rows, and the last 2 rows let you see if it really understands new cases.

### Feature Scaling

Feature scaling is about making sure all numbers are on a similar level. This is important because some features could have much bigger numbers than others and might unfairly influence the model.

**Example:**  
Suppose you have these two features:
- Age: ranges from 20 to 60
- Income: ranges from 30,000 to 120,000

With these values, “income” might dominate “age” in the way your model makes predictions.

**Popular Scaling Methods:**
- **Standardization:** Moves data so it’s centered around 0, with a typical spread of 1.
  - Example: Income mean is 60,000, standard deviation is 15,000. Income of 75,000 becomes:
    $$
    \text{Standardized} = \frac{75,000 - 60,000}{15,000} = 1
    $$
- **Min-Max Scaling:** Shrinks everything to be between 0 and 1.
  - Example: Age min is 20, max is 60. Age of 40 becomes:
    $$
    \text{Min-Max} = \frac{40 - 20}{60 - 20} = 0.5
    $$

### Why Scale After Splitting?

Scaling should **always be done after** splitting your dataset. If you scale before splitting, your test data influences the scaling, and this “leaks” information from the test set into training, which is unfair—like seeing test questions in advance.

**Correct Way (with Example):**
1. **Split data** into training and test sets.
2. **Fit scaler** only on training data.
   - If average income in training is 60,000, standard deviation is 15,000, use those numbers.
3. **Apply scaler** (using the values from training) to both training and test sets.

**Wrong Way (what to avoid):**
- Scale all data first, then split. This lets test data influence scaling, which could give you better test results than you’d get in real life.

### Key Takeaways

- Always split data before scaling or other preprocessing.
- Scale features after splitting—never before—to avoid unfair advantages (data leakage).
- Fit your scaler on the training set only, then use it to scale both sets.
- Scaling puts all features on an even playing field, making the model learn better.
- The whole reason for the test set is to measure real performance—keep it separate!

  # Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example data (rows: samples, columns: features)
# Let's imagine columns are: Age, Income
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

# 1. Split the dataset into training and test sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 2. Create the scaler (choose one: StandardScaler or MinMaxScaler)
scaler = StandardScaler()  # or use MinMaxScaler()

# 3. Fit the scaler only on the training data
scaler.fit(train_data)

# 4. Transform BOTH training and test data using the scaler
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# Print to see the results
print("Original training data:\n", train_data)
print("\nScaled training data:\n", train_scaled)
print("\nOriginal test data:\n", test_data)
print("\nScaled test data:\n", test_scaled)


***

If you’d like, I can add code snippets or even graphical visualizations to make these examples even clearer. Let me know if you want code!
