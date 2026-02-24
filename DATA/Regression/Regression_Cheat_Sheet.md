# Regression Cheat Sheet - Quick Reference

## 1. Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 2. Load & Explore
```python
# Load data
df = pd.read_csv('car data.csv')

# Quick exploration
df.head()          # First 5 rows
df.shape          # Rows, columns
df.info()         # Data types and missing values
df.describe()     # Statistical summary
df.isnull().sum() # Missing values count
```

---

## 3. Data Preprocessing

### Encode Categorical Variables
```python
# Label Encoding (for ordinal data)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])

# One-Hot Encoding (for nominal data)
df = pd.get_dummies(df, columns=['column'], drop_first=True)
```

### Separate Features & Target
```python
X = df.drop('Selling_Price', axis=1)  # Features
y = df['Selling_Price']                # Target
```

---

## 4. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% test, 80% train
    random_state=42     # For reproducibility
)
```

---

## 5. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 6. Model Training

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### Ridge Regression (L2 Regularization)
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)
```

### Lasso Regression (L1 Regularization)
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.01)
model.fit(X_train_scaled, y_train)
```

### Random Forest (No scaling needed)
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 7. Evaluation Metrics

### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
```
**Interpretation**: Average error magnitude (same units as target)

### Root Mean Squared Error (RMSE)
```python
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test, y_pred))
```
**Interpretation**: Average error (same units), penalizes large errors more

### R² Score
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```
**Interpretation**: Proportion of variance explained (0-1, higher is better)

### Mean Absolute Percentage Error (MAPE)
```python
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
```
**Interpretation**: Average percentage error

---

## 8. Quick Model Comparison
```python
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R²': r2}

# Evaluate multiple models
results = []
results.append(evaluate_model(y_test, model1.predict(...), 'Linear Reg'))
results.append(evaluate_model(y_test, model2.predict(...), 'Ridge'))
results.append(evaluate_model(y_test, model3.predict(...), 'Random Forest'))

results_df = pd.DataFrame(results)
print(results_df)
```

---

## 9. Visualizations

### Actual vs Predicted
```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
```

### Residual Plot
```python
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()
```

### Feature Importance (Tree-based models)
```python
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.barh(importance['Feature'], importance['Importance'])
plt.show()
```

---

## 10. Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'max_depth': [5, 10, 15, 20]
}

grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
```

---

## 11. Save & Load Models
```python
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

---

## 12. Make Predictions on New Data
```python
# Create new data with same features as training
new_data = pd.DataFrame({
    'Year': [2020],
    'Present_Price': [10.5],
    'Kms_Driven': [50000],
    # ... other features
})

# Scale if model requires it
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
print(f"Predicted Price: {prediction[0]:.2f}")
```

---

## 13. Model Selection Guide

| Model | Use When | Pros | Cons |
|-------|----------|------|------|
| **Linear Regression** | Linear relationship expected | Simple, interpretable | Assumes linearity |
| **Ridge** | High multicollinearity | Prevents overfitting | Slower than linear |
| **Lasso** | Need feature selection | Automatic feature selection | May be unstable |
| **Random Forest** | Non-linear patterns | Robust, feature importance | Black box |
| **Gradient Boosting** | Need best performance | Most accurate | Complex, slow |

---

## 14. Common Issues & Solutions

### Overfitting
- **Problem**: High training R², low test R²
- **Solutions**: 
  - Use Ridge/Lasso with regularization
  - Reduce model complexity
  - Use more training data

### Underfitting
- **Problem**: Low both training and test R²
- **Solutions**:
  - Increase model complexity
  - Add more features
  - Reduce regularization (alpha)

### Poor Predictions
- **Check**:
  - Missing values in features?
  - Features scaled properly?
  - Categorical variables encoded?
  - Enough training data?
  - Right model for the problem?

---

## 15. Complete Minimal Example
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load & preprocess
df = pd.read_csv('car data.csv')
le = LabelEncoder()
df['Transmission'] = le.fit_transform(df['Transmission'])

# Prepare data
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

## Evaluation Metrics Decision Tree

```
Am I trying to predict continuous values?
├─ YES → Regression
│  └─ How important are large errors?
│     ├─ Very Important → Use RMSE
│     ├─ Moderate → Use MAE
│     └─ Want % error → Use MAPE
│
└─ NO → Classification (different metrics)
```

---

## Remember!
- Always split data BEFORE scaling
- Scale training and test with SAME scaler
- Categorical variables must be encoded
- Evaluate on TEST set, not training set
- Compare multiple models to find the best
- Visualize predictions and residuals
- Domain knowledge is as important as model performance!

