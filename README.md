# 🔍 Electricity Consumption Prediction using Supervised Learning Models

This repository provides code and description of methodology along with the dataset.csv file for predicting monthly electricity consumption across 48 U.S. states using a range of supervised learning models.

### 📁 Dataset
The dataset used in this project is sourced from six key U.S. government agencies and has been meticulously merged to enable comprehensive analysis. It encompasses a wide range of data points, including energy consumption patterns, economic indicators, weather data, demographics, and environmental factors.

For a detailed description of the dataset sources and methodology, please refer to the paper titled "Electricity Demand Prediction Using Data-Driven Models: A Comprehensive Multi-Sector Analysis of Energy Consumption Dynamics."
- **Timeframe**: 1990–2023
- **Granularity**: Monthly observations
- **Coverage**: 48 U.S. states

### ✅ Preprocessing
- Missing values handled during data preparation.
- Feature scaling using StandardScaler:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
```
### ⚙️ Models & Hyperparameter Tuning
All models were trained using an 80/20 train-test split and optimized via GridSearchCV (5-fold cross-validation) to minimize Mean Squared Error (MSE).

#### 1. 🌲 Random Forest Regression (RF)

A non-linear ensemble model with internal feature importance metrics.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [300, 500, 1000],
    'max_features': ['sqrt'],
    'max_depth': [None, 3, 5],
    'min_samples_split': [1, 2, 3],
    'min_samples_leaf': [1, 2, 3]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(train_features, train_labels)
best_rf = grid_search.best_estimator_
rf_predictions = best_rf.predict(test_features)

# Feature importance
importances = best_rf.feature_importances_
feature_importances = sorted(zip(feature_list, importances), key=lambda x: x[1], reverse=True)[:10]

for feature, score in feature_importances:
    print(f"{feature}: {score:.5f}")
```
#### 2. ⚡ XGBoost Regression
A high-performance boosting model optimized for tabular data.

```python
import xgboost as xgb

param_grid = {
    'n_estimators': [500, 800, 1000],
    'max_depth': [1, 2, 3],
    'learning_rate': [0.01, 0.03, 0.05],
    'booster': ['gbtree', 'dart']
}

xgb_model = xgb.XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(train_features, train_labels)
xgb_model = grid_search.best_estimator_
xgb_predictions = xgb_model.predict(test_features)
```
#### 3. 📈 Support Vector Regression (SVR)
Effective for high-dimensional and sparse feature spaces.

```python
from sklearn.svm import SVR

param_grid = {
    'C': [100, 500, 1000],
    'gamma': [0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

svr_model = SVR()
grid_search = GridSearchCV(svr_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(train_features, train_labels)
svr_model = grid_search.best_estimator_
svr_predictions = svr_model.predict(test_features)
```
#### 4. 🔮 Gaussian Process Regression (GPR)
Bayesian model for modeling nonlinear trends.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

param_grid = {
    'alpha': [1e-2, 1e-5, 1e-10],
    'kernel': [ConstantKernel(1.0) * RBF(length_scale=l) for l in [0.5, 1.0, 2.0]],
    'n_restarts_optimizer': [0, 3, 5]
}

gpr_model = GaussianProcessRegressor(normalize_y=True, random_state=42)
grid_search = GridSearchCV(gpr_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(train_features, train_labels)
gpr_model = grid_search.best_estimator_
gpr_predictions = gpr_model.predict(test_features)
```
#### 5. 📉 NGBoost (Natural Gradient Boosting)
Probabilistic boosting model that provides flexible approximation.

```python
from ngboost import NGBRegressor
from ngboost.distns import Normal

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'minibatch_frac': [0.5, 1.0]
}

ngb_model = NGBRegressor(Dist=Normal, random_state=42)
grid_search = GridSearchCV(ngb_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(train_features, train_labels)
ngb_model = grid_search.best_estimator_
ngb_predictions = ngb_model.predict(test_features)
```
#### 6. 👥 K-Nearest Neighbors Regression (KNN)
Non-parametric approach that uses similarity between observations.

```python
from sklearn.neighbors import KNeighborsRegressor

param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3]
}

knn_model = KNeighborsRegressor()
grid_search = GridSearchCV(knn_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(train_features, train_labels)
knn_model = grid_search.best_estimator_
knn_predictions = knn_model.predict(test_features)
```
#### 7. ➕ Multilinear Regression (MLR)
Baseline linear model for comparison.

```python
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression().fit(train_features, train_labels)
lr_predictions = lr_model.predict(test_features)
```
### 📊 Model Evaluation
Standard regression metrics were used to evaluate each model:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': MAPE(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

results = {
    "Random Forest": evaluate(test_labels, rf_predictions),
    "XGBoost": evaluate(test_labels, xgb_predictions),
    "SVR": evaluate(test_labels, svr_predictions),
    "GPR": evaluate(test_labels, gpr_predictions),
    "NGBoost": evaluate(test_labels, ngb_predictions),
    "KNN": evaluate(test_labels, knn_predictions),
    "Linear Regression": evaluate(test_labels, lr_predictions)
}
```
### 📈 Metrics Reported:
| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |
| **R²** | Coefficient of Determination |
