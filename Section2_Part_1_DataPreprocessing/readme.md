# 🧠 Machine Learning Workflow & Feature Scaling
---
## 1️⃣ Machine Learning Process Overview

Machine learning projects typically follow a **three-step process**:

1. **Data Pre-processing**  
   - Import, clean, and prepare your data  
   - **Split dataset into training and test sets**  
     - Ensures model evaluation is based on **unseen data**  
     - Provides an unbiased measure of how well the model generalizes  

2. **Modeling**  
   - Build, train, and make predictions using machine learning algorithms

3. **Evaluation**  
   - Assess model performance using appropriate metrics  

> 💡 **Tip:** Practicing these steps hands-on helps you understand how ML models are built, trained, and assessed in real-world applications.

### 🗝️ Key Takeaways

- ML process steps: **Data Pre-processing → Modeling → Evaluation**  
- Proper **data splitting** ensures unbiased evaluation and better generalization  

---

## 2️⃣ Data Preprocessing: Importance of the Training-Test Split

### Overview
Splitting a dataset into **training and test sets** is one of the most important steps in the machine learning workflow.  
This ensures that model evaluation is based on **unseen data**, providing an unbiased measure of how well the model generalizes beyond the examples it was trained on.

### Summary
- Splitting your data into **training and test sets** is critical for reliable model assessment.  
- Allows you to measure how well your model **generalizes**, rather than just memorizing training examples.

### 🗝️ Key Takeaways

- The **training-test split** provides an unbiased way to evaluate model performance  
- Common split ratio: **80% training / 20% testing**  
- The test set must remain **unseen during training** to simulate real-world conditions  
- Comparing **predicted vs actual outcomes** on the test set helps identify how accurate and generalizable the model is  

---

## 3️⃣ Feature Scaling

Feature scaling is a **crucial preprocessing step** in machine learning.

- Always applied to **columns**, never rows  
- Each feature column is scaled independently to ensure **consistent ranges and units**

> ⚠️ **Note:** Comparing features with different units or magnitudes without scaling can bias model performance.

### 🔧 Types of Feature Scaling

#### 🌟 Normalization

- Rescales data to a **0 to 1 range**  
- **Formula:**  
```text
Normalized value = (value - min) / (max - min)
````

#### 🌟 Standardization

* Centers data by subtracting the **mean** and scaling by the **standard deviation**
* Most values lie between **-3 and 3** (outliers can go beyond)
* **Formula:**

```text
Standardized value = (value - mean) / std
```

### 🗝️ Key Takeaways for Feature Scaling

* ⚡ Always applied to **columns**, never rows
* 🔹 **Normalization:** rescales data to 0–1
* 🔹 **Standardization:** rescales data to mean 0 and standard deviation 1 (approx. -3 to 3)
* ⚡ Ensures fair comparison between features with different units or magnitudes

---

## 4️⃣ ✅ Conclusion

* The ML process is an **iterative, structured approach**:
  **Data Pre-processing → Modeling → Evaluation**
* Proper **data splitting** and **feature scaling** ensure unbiased evaluation and fair contribution of all features
* Practicing these steps hands-on improves understanding of **how ML models are built, trained, and evaluated**
