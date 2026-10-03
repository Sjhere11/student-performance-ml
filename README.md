# Student Performance ML

A practical machine learning project using a 500-row student performance dataset to practice three scikit-learn algorithms:

- Linear Regression
- Logistic Regression
- Decision Tree Classifier

## Project Structure

```text
student-performance-ml/
│
├── data/
│   └── student_performance_ml_500.csv
│
├── linear_regression.py
├── logistic_regression.py
├── decision_tree.py
├── README.md
└── requirements.txt
```

## Dataset

The dataset contains 500 student records and includes:

- `hours_studied`
- `attendance`
- `previous_score`
- `assignment_completed`
- `sleep_hours`
- `extracurricular`
- `internet_quality`
- `final_score`
- `passed`

`student_id` is an identifier and is not used as a model feature.

## Machine Learning Tasks

### 1. Linear Regression

**Goal:** predict the student's `final_score`.

Evaluation metrics:
- MAE
- MSE
- RMSE
- R²

Run:

```bash
python linear_regression.py
```

### 2. Logistic Regression

**Goal:** predict whether the student `passed`.

Target:
- `0` = Fail
- `1` = Pass

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Run:

```bash
python logistic_regression.py
```

### 3. Decision Tree

**Goal:** predict whether the student `passed`.

The Decision Tree uses the same classification task as Logistic Regression, allowing the two classifiers to be compared directly.

Run:

```bash
python decision_tree.py
```

## Preprocessing

- `internet_quality` is categorical and is converted using one-hot encoding.
- Numeric features are standardized for Logistic Regression.
- Decision Trees do not require feature scaling.

## Results

The current dataset produced approximately:

| Model | Task | Results |
|---|---|---|
| Linear Regression | Predict final score | RMSE: 5.1532, R²: 0.7585 |
| Logistic Regression | Predict pass/fail | Accuracy: 0.9800, F1: 0.9897 |
| Decision Tree | Predict pass/fail | Accuracy: 0.9700, F1: 0.9846 |

These results are based on an 80/20 train-test split with `random_state=42`.

## Installation

Create a virtual environment if desired:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## What I Practiced

- Loading CSV data with pandas
- Separating features and targets
- Train/test splitting
- Categorical feature encoding
- Feature scaling
- Linear Regression
- Logistic Regression
- Decision Trees
- Model predictions
- Regression metrics
- Classification metrics
- Confusion matrices
- Comparing classification models

## Author

Built as a practical machine learning portfolio project while learning scikit-learn.
