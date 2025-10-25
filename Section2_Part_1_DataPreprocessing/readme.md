# 🧠 Machine Learning Workflow & Feature Scaling
---

## 1️⃣ Machine Learning Process Overview

Machine learning projects typically follow a **three-step process**:

1. **Data Pre-processing**  
   - Import, clean, and split your data  
   - Splitting into **training and test sets** ensures model evaluation on **unseen data**, providing an unbiased measure of generalization

2. **Modeling**  
   - Build, train, and make predictions using machine learning algorithms

3. **Evaluation**  
   - Assess model performance using appropriate metrics  

> 💡 **Tip:** Practicing these steps hands-on helps you understand how ML models are built, trained, and assessed in real-world applications.

### 🗝️ Key Takeaways

- ML process steps: **Data Pre-processing → Modeling → Evaluation**  
- Proper data splitting ensures unbiased evaluation  

---

## 2️⃣ Feature Scaling

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

**Calculating Standard Deviation (std):**

1. Compute the mean of the column:
   [
   mean = \frac{\sum x_i}{n}
   ]
2. Calculate squared differences from the mean:
   [
   (x_i - mean)^2
   ]
3. Compute variance:
   [
   variance = \frac{\sum (x_i - mean)^2}{n} \quad \text{or} \quad \frac{\sum (x_i - mean)^2}{n-1}
   ]
4. Take the square root:
   [
   std = \sqrt{variance}
   ]

---

### 🗝️ Key Takeaways for Feature Scaling

* ⚡ Always applied to **columns**, never rows
* 🔹 **Normalization:** rescales data to 0–1
* 🔹 **Standardization:** rescales data to mean 0 and standard deviation 1 (approx. -3 to 3)
* ⚡ Ensures fair comparison between features with different units or magnitudes

---

## 3️⃣ ✅ Conclusion

* The ML process is an **iterative, structured approach**:
  **Data Pre-processing → Modeling → Evaluation**
* Proper feature scaling ensures that **all features contribute appropriately** to similarity measures and model performance
* Practicing these steps hands-on improves understanding of **how ML models are built, trained, and evaluated**

```
```
