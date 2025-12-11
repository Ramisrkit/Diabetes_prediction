

---

# 🩺 Diabetes Prediction Using Glucose & Blood Pressure

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-84%25-brightgreen)]()
[![Dataset Size](https://img.shields.io/badge/Dataset-768%20rows-orange)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--12--11-blueviolet)]()
[![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-FF5733?logo=googlecolab\&logoColor=white)](https://colab.research.google.com/drive/<your-colab-link>)

Predict diabetes risk using only **Glucose** and **BloodPressure**. Lightweight, interpretable, and effective for early screening.

---

## 🚀 Project Overview

Diabetes is a chronic condition that can lead to severe complications if undetected. This project demonstrates **how a simple machine learning model can achieve high accuracy using just two features**:

* **Glucose** – Blood sugar level after 2 hours
* **BloodPressure** – Diastolic blood pressure

Despite the minimal feature set, the model achieves **84% accuracy** on both training and test sets.

> **Goal:** Create a small, interpretable model suitable for educational or clinical screening purposes.

---

## 📊 Dataset

| Feature       | Description                          |
| ------------- | ------------------------------------ |
| Glucose       | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg)     |
| Outcome       | Diabetes presence (0 = No, 1 = Yes)  |

> Source: [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🛠️ Model

* **Algorithm:** Decision Tree Classifier
* **Reason:** Interpretable and effective for small feature sets

### Training Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('diabetes.csv')
X = df[['Glucose', 'BloodPressure']]
y = df['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))
```

---

## 📈 Performance

| Metric            | Score |
| ----------------- | ----- |
| Training Accuracy | 0.84  |
| Testing Accuracy  | 0.84  |

> Strong generalization with minimal features.

---

## 📊 Visualizations

### Feature Distribution

![Glucose vs Outcome](https://user-images.githubusercontent.com/yourusername/glucose_vs_outcome.png)
*Scatter plot showing Glucose levels vs Diabetes outcome*

### Decision Boundary

![Decision Boundary](https://user-images.githubusercontent.com/yourusername/decision_boundary.png)
*Decision Tree boundary visualization with Glucose and BloodPressure*

### Real-Time Prediction GIF

![Prediction Example](https://user-images.githubusercontent.com/yourusername/prediction_example.gif)
*Animated GIF showing the model predicting diabetes outcomes for multiple patients in real-time*

---

## 💡 Insights

* **Glucose** is the strongest predictor of diabetes.
* **BloodPressure** contributes moderately to prediction.
* Minimalistic models can be **lightweight, interpretable, and effective**.
* Decision Trees provide **clear decision paths**, ideal for teaching or clinical use.

---

## ⚡ Usage

1. Clone the repository:

```bash
git clone <repo-url>
cd diabetes-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the prediction script:

```bash
python diabetes_prediction.py
```

4. Modify **Glucose** and **BloodPressure** values in the script to predict outcomes for new patients.

5. **Try it online on Colab:** [Open in Google Colab](https://colab.research.google.com/drive/<your-colab-link>)

---

## 📂 Folder Structure

```
diabetes-prediction/
│
├─ diabetes.csv
├─ diabetes_prediction.py
├─ README.md
├─ requirements.txt
├─ visualizations/
│   ├─ glucose_vs_outcome.png
│   ├─ decision_boundary.png
│   └─ prediction_example.gif
└─ LICENSE
```

---

## 📌 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---


