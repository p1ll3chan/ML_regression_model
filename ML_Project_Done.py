#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/p1ll3chan/ML_regression_model/blob/main/ML_Project_Done.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install --upgrade --force-reinstall numpy==1.26.4 scikit-learn==1.3.2 joblib')


# In[ ]:


pip install numpy==1.26.4 --force-reinstall


# In[9]:


# Cell 1 - Imports
# (Colab usually has these; run to ensure everything is available)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


# Cell 2 - Upload dataset manually (Colab)
from google.colab import files
uploaded = files.upload()   # Click "Choose Files" and pick your auto-mpg.csv

# After upload, read the CSV (replace filename if different)
import io
fname = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[fname]))
print(f"Loaded: {fname}  — shape: {df.shape}")
df.head()


# In[11]:


# Cell 3 - Clean & convert target
# Rename columns to consistent names (handles variations)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# Convert mpg to km per liter
conv_factor = 1.609344 / 3.785411784   # exact conversion factor
df['kml'] = df['mpg'] * conv_factor

# Drop car name (we decided not to use it)
if 'car_name' in df.columns:
    df = df.drop(columns=['car_name'])
elif 'car name' in df.columns:
    df = df.drop(columns=['car name'])

# Ensure horsepower numeric if needed
if 'horsepower' in df.columns:
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Show cleaned head and info
print("After cleaning, columns:\n", df.columns.tolist())
df.describe().T


# In[12]:


# Cell 4 - Features and split
features = ['cylinders','displacement','horsepower','weight','acceleration','model_year','origin']
target = 'kml'

# If some feature names differ (e.g., 'model year'), try to fix
if 'model_year' not in df.columns and 'model_year' not in features:
    # try alternate column name
    for alt in ['model_year','model year','year']:
        if alt in df.columns:
            df = df.rename(columns={alt:'model_year'})
            break

X = df[features].copy()
y = df[target].copy()

# Drop rows with NA in features/target
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# In[14]:


# Cell 5 - Preprocessing
numeric_features = ['cylinders','displacement','horsepower','weight','acceleration','model_year']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

categorical_features = ['origin']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Changed sparse to sparse_output
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# In[15]:


# Cell 6 - Models dictionary
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01, max_iter=5000),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42)
}

# Train each model, evaluate, and save pipelines
results = []
trained_pipes = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('pre', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append({'model': name, 'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse})
    trained_pipes[name] = pipe
    # Optional: save model to disk in Colab runtime
    joblib.dump(pipe, f"{name}_pipeline.joblib")

results_df = pd.DataFrame(results).sort_values('rmse')
results_df


# In[16]:


# Cell 7 - Display comparison with nicer formatting
results_df[['model','r2','mae','rmse']].style.format({
    'r2': '{:.4f}',
    'mae': '{:.4f}',
    'rmse': '{:.4f}'
})


# In[17]:


# Cell 8 - Feature importance for RandomForest
best_rf = trained_pipes.get('RandomForest')
if best_rf is not None:
    # Extract feature names produced by ColumnTransformer
    # numeric then onehot origin columns
    num_cols = numeric_features
    # get onehot feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_cols = [f"origin_{int(x)}" for x in ohe.categories_[0]]
    feature_names = num_cols + ohe_cols
    # get importance from RandomForest model inside pipeline
    rf_model = best_rf.named_steps['model']
    importances = rf_model.feature_importances_
    fi = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    print(fi)
    # Plot
    plt.figure(figsize=(8,5))
    plt.barh(fi['feature'], fi['importance'])
    plt.gca().invert_yaxis()
    plt.title('Random Forest Feature Importances')
    plt.xlabel('Importance')
    plt.show()
else:
    print("RandomForest model not found.")


# In[18]:


# Cell 9 - Choose best model by RMSE and plot actual vs predicted
best_model_name = results_df.iloc[0]['model']
print("Best model by RMSE:", best_model_name)
best_pipe = trained_pipes[best_model_name]
y_pred = best_pipe.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1)
plt.xlabel('Actual KM/L')
plt.ylabel('Predicted KM/L')
plt.title(f'Actual vs Predicted — {best_model_name}')
plt.grid(True)
plt.show()

# Print a small error summary for the best model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"{best_model_name}  —  R2: {r2:.4f}, MAE: {mae:.4f} km/l, RMSE: {rmse:.4f} km/l")


# In[19]:


# Cell 10 - Save results table to CSV in Colab VM (downloadable)
results_df.to_csv('model_results_kml.csv', index=False)
print("Saved model_results_kml.csv — download from Colab Files panel if needed.")

