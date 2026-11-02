# Customer Churn & Retention Analytics

## Project Overview
This project uses Python and Machine Learning to analyze customer churn and predict whether a telecom customer is likely to leave the company.

The goal is to identify churn patterns and help businesses understand customer retention risks.

## Objectives
- Analyze customer churn patterns.
- Clean and preprocess customer data.
- Explore important churn-related factors.
- Build a Logistic Regression classification model.
- Predict churn probability for new customers.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Joblib
- VS Code

## Project Workflow
1. Data Analysis and Visualization
2. Data Cleaning and Preprocessing
3. Feature Encoding and Scaling
4. Train-Test Split
5. Logistic Regression Model Training
6. Model Evaluation
7. Customer Churn Prediction

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 73.81% |
| ROC-AUC | 0.8416 |
| Churn Recall | 78% |
| Churn Precision | 50% |

The model uses class weighting to improve the identification of customers who may churn.

## Key Findings
- Customer tenure is an important predictive feature.
- Month-to-month contracts are associated with higher churn risk.
- Two-year contracts are associated with lower churn risk.
- Fiber optic internet customers show higher churn in this dataset.

These are model associations, not proof of causation.

## Project Files
- `churn_cleaning.py` — Data cleaning
- `churn_ml.py` — Model training and evaluation
- `churn_prediction.py` — New customer prediction
- `cleaned_churn.csv` — Cleaned dataset
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — Original dataset
- `churn_model.pkl` — Saved machine learning pipeline
- `.gitignore` — Files excluded from Git

## How to Run

Clone the repository:

```bash
git clone https://github.com/nasirhussain0419-arch/customer-churn-retention-analytics.git
```

Open the project folder:

```bash
cd customer-churn-retention-analytics
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Run the prediction:

```bash
python churn_prediction.py
```

## Future Improvements
- Experiment with Random Forest and other classifiers.
- Tune model hyperparameters.
- Improve churn prediction precision.
- Build an interactive dashboard.
- Develop a customer retention recommendation system.

## Author
Nasir Hussain

GitHub: https://github.com/nasirhussain0419-arch

## Disclaimer
This project is created for educational and portfolio purposes. Predictions are estimates and should not be treated as guaranteed customer behaviour.
