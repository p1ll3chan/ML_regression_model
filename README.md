# ML Regression Model

A machine learning project demonstrating the implementation, training, and evaluation of multiple regression models using a real-world dataset.

---

## Overview

This repository showcases a complete regression workflow including data preprocessing, model training, performance evaluation, and comparison across different regression algorithms.

The project is intended for learning, experimentation, and as a reference for regression-based ML pipelines.

---

## Project Structure
```
├── ML_Project_Done.ipynb # Jupyter notebook with full workflow
├── ML_Project_Done.py # Python script for model training and evaluation
├── auto-mpg.csv # Dataset
├── model_results_kml.csv # Model evaluation results
└── README.md
```


---

## Models Implemented

- Linear Regression  
- Polynomial Regression  
- Ridge Regression  
- Lasso Regression  
- Support Vector Regression (SVR)  
- Decision Tree Regression  
- Random Forest Regression  
- MLP Regressor (Neural Network)

---

## Workflow

1. Load and preprocess the dataset  
2. Perform feature scaling and train-test split  
3. Train multiple regression models  
4. Evaluate models using standard regression metrics  
5. Store and compare results

---

## Evaluation Metrics

- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

Evaluation results are saved in `model_results_kml.csv`.

---

## Requirements

1. Python 3.x with the following libraries:
2. numpy
3. pandas
4. scikit-learn
5. matplotlib
6. seaborn

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn

