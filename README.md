# 💼 AI Job Salary Prediction

A machine-learning regression project that predicts AI job salaries using job, experience, company, location, employment, and other job-related attributes.

## 📌 Project Overview

This project builds and compares multiple regression models for predicting `salary_usd`.

The workflow covers:

**Data collection → Data inspection → Cleaning → EDA → Preprocessing → Model training → Hyperparameter tuning → Model comparison → Final evaluation → Model persistence**

The dataset contains **15,000 records and 19 columns**.

## 🎯 Objectives

- Understand salary patterns in AI-related job postings.
- Identify factors associated with salary.
- Prepare categorical and numerical features for machine learning.
- Compare multiple regression algorithms.
- Tune model hyperparameters.
- Evaluate the final model using regression metrics.
- Save the trained model and preprocessing pipeline for later use.

## 📊 Dataset

**Records:** 15,000  
**Features:** 19

Important fields include:

- `job_title`
- `salary_usd`
- `salary_currency`
- `experience_level`
- `employment_type`
- `company_location`
- `company_size`
- `employee_residence`
- `remote_ratio`
- `required_skills`
- `education_required`
- `years_experience`
- `industry`
- `posting_date`
- `application_deadline`
- `job_description_length`
- `benefits_score`
- `company_name`

Target variable:

```text
salary_usd
```

The notebook reports no missing values and no duplicate records in the dataset.

## 🔎 Exploratory Data Analysis

The project examines:
- Numerical feature distributions
- Job-title distributions
- Experience-level categories
- Company-size categories
- Salary distribution
- Relationships between job characteristics and salary

## 🧹 Preprocessing

The project removes the identifier column:

```text
job_id
```

Categorical and numerical features are prepared for machine-learning models using preprocessing techniques implemented in the notebook.

## 🤖 Models Compared

The notebook evaluates:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. KNeighbors Regressor
6. Support Vector Regression

### Initial Model Comparison

| Model | Test R² | Test RMSE |
|---|---:|---:|
| Linear Regression | 0.01 | 46,877.25 |
| Decision Tree | 0.83 | 18,098.59 |
| Random Forest | 0.88 | 15,173.19 |
| Gradient Boosting | 0.88 | 14,754.43 |
| KNN Regressor | 0.74 | 21,708.59 |
| SVR | 0.43 | 30,824.96 |

## ⚙️ Hyperparameter Tuning

The notebook tunes several models.

### Decision Tree

```text
max_depth = 15
min_samples_leaf = 5
min_samples_split = 2
```

### Gradient Boosting

```text
learning_rate = 0.1
max_depth = 5
n_estimators = 200
```

### Random Forest

```text
max_depth = 15
min_samples_split = 10
n_estimators = 200
```

### KNN

```text
n_neighbors = 7
weights = distance
```

### SVR

```text
C = 100
epsilon = 0.1
```

## 🌲 Final Model

The notebook saves a **Random Forest Regressor** as the final model.

Final hyperparameters:

```text
n_estimators = 200
max_depth = 15
min_samples_split = 10
```

## 📈 Final Evaluation

The final Random Forest model achieved:

| Metric | Training | Testing |
|---|---:|---:|
| MAE | 10,456.89 | 14,978.66 |
| MSE | 195,973,446.67 | 435,665,468.77 |
| RMSE | 13,999.05 | 20,872.60 |
| R² | 0.9460 | 0.8805 |

The notebook's feature-importance analysis identifies experience-related features, including `experience_level` and `years_experience`, among the most influential features.

## 💾 Model Persistence

The project saves:

```text
random_forest_salary_model.pkl
preprocessor.pkl
```

The notebook also demonstrates loading the saved model and preprocessor back into Python.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Regression
- Hyperparameter tuning
- Pickle

## 📂 Project Structure

```text
AI-Job-Salary-Prediction/
│
├── AI job Salary Prediction -ML Project(2).ipynb
├── ai_job_dataset.csv
├── random_forest_salary_model.pkl
├── preprocessor.pkl
└── README.md
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open the project notebook and update the dataset path if required.

## 💡 Applications

- Salary estimation for AI/ML roles
- Compensation analysis
- Job-market analytics
- Recruiter decision support
- Understanding salary-related job factors

## 🚀 Future Improvements

- Build a Streamlit salary prediction application.
- Add explainability using SHAP.
- Experiment with XGBoost/LightGBM.
- Add cross-validation and robust error analysis.
- Create salary prediction ranges instead of only point estimates.

## 👩‍💻 Author

**Challa Swapna**

GitHub: `https://github.com/swapnachalla4826-sudo`
