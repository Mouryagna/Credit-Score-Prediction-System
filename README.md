# Credit Score Prediction System

Predicting credit score categories using machine learning with a modular end-to-end pipeline and a simple Flask-based prediction interface.

---

## Overview

This project implements an end-to-end machine learning workflow to classify a person's **credit score** based on financial and credit behavior.

The system includes:

- Data ingestion
- Data preprocessing
- Model training
- Model evaluation
- Model serialization
- A simple Flask web interface for predictions

The main goal of this project is to demonstrate a **structured machine learning pipeline**.  
The web interface is intentionally simple and is only used to test predictions.

---

## Problem Statement

Credit scoring is widely used by financial institutions to determine a person's **creditworthiness**.

This project builds a machine learning model that predicts the credit score category:

- Good
- Standard
- Poor

based on financial behavior and credit history.

---

## Features Used

Example features used in the dataset:

- Month
- Age
- Num_Bank_Accounts
- Num_Credit_Card
- Interest_Rate
- Delay_from_due_date
- Num_of_Delayed_Payment
- Changed_Credit_Limit
- Num_Credit_Inquiries
- Credit_Utilization_Ratio
- Monthly_Balance
- Credit_History_Age
- Investment_Ratio
- Num_Loan_Types
- Debt_to_Income
- EMI_to_Income
- Delay_Intensity
- Utilization_Delay
- Debt_per_Loan

Categorical features:

- Occupation
- Credit_Mix
- Payment_of_Min_Amount
- Spending_Level
- Payment_Value_Level

Target variable:

Credit_Score

---

## Machine Learning Pipeline

The project follows a modular machine learning pipeline:

Dataset  
↓  
Data Ingestion  
↓  
Data Transformation  
↓  
Model Training  
↓  
Model Selection  
↓  
Saved Model  
↓  
Prediction Pipeline  
↓  
Flask Web Interface

---

## Models Used

Several machine learning models were trained and compared.

Examples:

- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

The best performing model is selected and saved.

---

## Project Structure

```
Credit-Score-Prediction-System

artifacts/
    model.pkl
    preprocessor.pkl
    label_encoder.pkl

notebook/
    data/
    EDA.ipynb
    model_training.ipynb

src/
    components/
        data_ingestion.py
        data_transformation.py
        model_trainer.py

    pipeline/
        train_pipeline.py
        predict_pipeline.py

    utils.py
    logger.py
    exception.py

templates/
    index.html
    home.html

static/

app.py
requirements.txt
README.md
```

---

## Web Interface

A simple Flask application is used to test predictions.

Users can:

1. Enter financial information
2. Submit the form
3. Receive predicted credit score

Example output:

Prediction: Good

---

## Installation

Clone the repository

```
git clone <https://github.com/Mouryagna/Credit-Score-Prediction-System>
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Project

Train the model

```
python -m src.pipeline.train_pipeline
```

Run the Flask application

```
python app.py
```

Open in browser

```
http://localhost:5000
```

---

## Future Improvements

Possible improvements:

- Better feature engineering
- Improved UI design
- Hyperparameter tuning
- Cloud deployment

---

## Author

Mouryagna  
AI & Machine Learning