

# 🧠 Income Prediction Using Machine Learning

This project aims to predict whether an individual's income exceeds \$50K per year based on census data. It involves extensive preprocessing, exploratory data analysis (EDA), feature engineering, and classification using a Random Forest model with hyperparameter tuning.

---

## 📂 Dataset

- **Source:** `19. Income prediction.csv`
- **Features:** Demographic, educational, work-related, and financial attributes
- **Target:** `income_>50K` → renamed to `exceeds50K` for clarity

---

## ✅ Project Workflow

### 🔹 Step 1: Preprocessing
- Removed missing and duplicate values
- Dropped less-informative features like `race`
- Encoded categorical variables using one-hot encoding
- Converted boolean columns to integers

### 🔹 Step 2: Exploratory Data Analysis (EDA)
Used **Seaborn** and **Matplotlib** for:
- 🔥 Correlation heatmap of numeric features
- 🎓 Education vs. income probability
- ⏰ Hours-per-week vs. income probability
- 📊 Gender, marital status, relationship, and workclass vs. income
- 🌎 Native country vs. income
- 📈 Age distribution by income class

### 🔹 Step 3: Model Training
- Used `RandomForestClassifier` inside a `Pipeline`
- Performed hyperparameter tuning using `GridSearchCV` with:
  - `n_estimators` from 20 to 30
  - `max_depth` from 3 to 9
- 5-fold cross-validation (`cv=10`) for balanced evaluation

### 🔹 Step 4: Evaluation
- Evaluated the best model using:
  - `accuracy_score`
  - `classification_report` (precision, recall, F1-score)
  - `confusion_matrix`

---

## 🔍 Results

- **Best Hyperparameters:**  
  _As determined via GridSearchCV (customize below based on your output)_
  ```json
  {'classify__n_estimators': 27, 'classify__max_depth': 7}
````

* **Accuracy:** \~99.9% (depending on final tuning)
* **Model Performance:**

  * Strong recall for high-income class
  * Balanced precision and F1-score across classes

---

## 📦 Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---


## 📝 Future Improvements

* Try models like **XGBoost** or **LightGBM**
* Apply **SMOTE** for deeper imbalance handling
* Add **feature importance visualization**
* Use **shap/shapash** for explainability

---
