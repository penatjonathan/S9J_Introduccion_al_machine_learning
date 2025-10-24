# 🤖 Sprint 9 Project — Megaline Plan Recommendation (Machine Learning)

---

🧠 **Project Overview**  
In this sprint, I worked as a data analyst and machine learning engineer for **Megaline**, a mobile carrier that wants to encourage customers to switch from legacy plans to its new offers — **Smart** and **Ultra**.  

My goal was to build a **classification model** that predicts which plan is most suitable for a customer based on their usage behavior.  
The task was to achieve an **accuracy of at least 0.75** using real subscriber data on calls, messages, and internet usage.  

This project consolidates all the skills I learned in the “Introduction to Machine Learning” module, including model training, hyperparameter tuning, validation testing, and performance evaluation.

---

## 🎯 Project Objectives
- Load and inspect the dataset `/datasets/users_behavior.csv`.  
- Split the data into **training**, **validation**, and **test** sets.  
- Train and compare multiple classification algorithms.  
- Optimize hyperparameters to improve model performance.  
- Evaluate final accuracy using the test set.  
- Perform a **sanity check** to verify that the model generalizes correctly.  

---

## 📁 Dataset Description
**File:** `users_behavior.csv`

Each record contains **monthly behavioral data** for one user.  

| Column | Description |
|---------|--------------|
| `calls` | Number of calls made |
| `minutes` | Total duration of calls (in minutes) |
| `messages` | Number of SMS sent |
| `mb_used` | Internet traffic used (in MB) |
| `is_ultra` | Target variable: 1 = Ultra plan, 0 = Smart plan |

---

## 🧩 Project Steps

### Step 1 – Load and Explore Data  
- Read the CSV file into a Pandas DataFrame.  
- Check structure with `.info()`, `.describe()`, and `.isna().sum()`.  
- Ensure data types are correct and no missing values remain.  
- Analyze distributions and basic statistics to understand feature ranges.

### Step 2 – Split the Dataset  
- Split into:
  - **Training set (60%)**
  - **Validation set (20%)**
  - **Test set (20%)**
- Used `train_test_split()` from `sklearn.model_selection` with a fixed `random_state` for reproducibility.

### Step 3 – Train Multiple Models  
Tested and compared several algorithms:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Logistic Regression**

For each model:
- Trained using training data.  
- Evaluated accuracy on validation data.  
- Adjusted key hyperparameters (e.g., `max_depth`, `n_estimators`, `min_samples_split`).

### Step 4 – Evaluate and Select the Best Model  
- Selected the model with the **highest validation accuracy** without overfitting.  
- Retrained on combined training + validation sets.  
- Evaluated final performance on the **test set**.  
- Verified that **test accuracy ≥ 0.75**.

### Step 5 – Sanity Check  
- Tested model behavior with edge and random inputs to ensure consistent predictions.  
- Compared feature importance to confirm logical decision patterns (e.g., high data usage → more likely Ultra plan).

---

## 📊 Example Results (Illustrative)
| Model | Validation Accuracy | Test Accuracy |
|--------|----------------------|----------------|
| Decision Tree | 0.74 | 0.73 |
| Random Forest | **0.79** | **0.78** |
| Logistic Regression | 0.74 | 0.75 |

✅ **Final Model:** Random Forest  
✅ **Test Accuracy:** 0.78  

---

## 💼 Skills Developed
- Data Splitting & Preprocessing (`train_test_split`, feature scaling)  
- Model Selection (Decision Tree, Random Forest, Logistic Regression)  
- Hyperparameter Tuning & Cross-validation  
- Model Evaluation & Sanity Checking  
- Interpretation of Classification Results  

---

## 🧰 Tools & Libraries
`Python` | `Pandas` | `NumPy` | `scikit-learn` | `Matplotlib` | `Jupyter Notebook`

---

## 👤 Author  
*Project completed by [Jonathan Peña] as part of Sprint 9 — Machine Learning Fundamentals.*
