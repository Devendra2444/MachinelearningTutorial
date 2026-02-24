# Step-by-Step Guide: Predicting Selling Price Using Regression

## Overview
This guide will walk you through building a regression model to predict car selling prices using machine learning.

---

## Step 1: Data Loading & Exploration

### What is it?
Loading and understanding your dataset before building a model.

### Code:
```python
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Load the car data
df = pd.read_csv('/home/deven/PycharmProjects/Ml Tutorial/car data.csv')

# Display first few rows
print(df.head(10))

# Check shape (rows, columns)
print(df.shape)

# Get basic statistics
print(df.describe())

# Check data types
print(df.info())

# Check for missing values
print(df.isnull().sum())
```

### What to Look For:
- **Shape**: Number of samples and features
- **Data Types**: Which columns are numeric, categorical, etc.
- **Missing Values**: Any null/NaN values that need handling
- **Statistics**: Mean, std, min, max values for numeric columns
- **Unique Values**: For categorical columns (Owner, Transmission, Fuel_Type, Seller_Type)

---

## Step 2: Data Preprocessing

### What is it?
Converting raw data into a format suitable for machine learning models.

### Key Tasks:

#### 2.1 Handle Categorical Variables (Encoding)
Categorical variables (text values) must be converted to numbers:

```python
# Check unique values in categorical columns
print(df['Transmission'].unique())  # Manual, Automatic
print(df['Fuel_Type'].unique())      # Petrol, Diesel, CNG
print(df['Seller_Type'].unique())    # Dealer, Individual

# Method 1: Label Encoding (for ordinal data like Transmission)
from sklearn.preprocessing import LabelEncoder

le_transmission = LabelEncoder()
df['Transmission'] = le_transmission.fit_transform(df['Transmission'])

le_fuel = LabelEncoder()
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])

le_seller = LabelEncoder()
df['Seller_Type'] = le_seller.fit_transform(df['Seller_Type'])

# Method 2: One-Hot Encoding (creates dummy variables)
df = pd.get_dummies(df, columns=['Fuel_Type'], drop_first=True)
df = pd.get_dummies(df, columns=['Seller_Type'], drop_first=True)
```

#### 2.2 Drop Irrelevant Columns
```python
# Remove Car_Name as it's just a label (not useful for prediction)
df = df.drop('Car_Name', axis=1)
```

#### 2.3 Separate Features (X) and Target (y)
```python
# Separate independent variables (X) and dependent variable (y)
X = df.drop('Selling_Price', axis=1)  # Features
y = df['Selling_Price']                # Target variable

print("Features shape:", X.shape)
print("Target shape:", y.shape)
```

---

## Step 3: Train-Test Split

### What is it?
Dividing data into training set (to train model) and test set (to evaluate model).

### Why?
- **Training Set (80%)**: Used to train the model
- **Test Set (20%)**: Used to evaluate how well the model performs on unseen data

### Code:
```python
from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
```

---

## Step 4: Feature Scaling (Normalization)

### What is it?
Bringing all features to a similar scale to improve model performance.

### Why?
Some features (like Kms_Driven) have large values, while others (Owner) have small values. This can bias the model.

### Code:
```python
from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data (use same scaler)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

---

## Step 5: Model Selection & Training

### What is it?
Choosing a regression algorithm and training it on your data.

### Common Regression Algorithms:

#### 5.1 Linear Regression (Simplest)
```python
from sklearn.linear_model import LinearRegression

# Create model
lr_model = LinearRegression()

# Train on training data
lr_model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print("Coefficients:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)
```

#### 5.2 Ridge Regression (Prevents Overfitting)
```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
```

#### 5.3 Lasso Regression (Feature Selection)
```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)
```

#### 5.4 Random Forest Regressor (Non-linear)
```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # No scaling needed for tree-based models
```

---

## Step 6: Model Evaluation

### What is it?
Assessing how well your model performs on unseen data.

### Key Metrics:

#### 6.1 Make Predictions
```python
# Predict on test set
y_pred = lr_model.predict(X_test_scaled)

print("First 10 predictions:", y_pred[:10])
print("First 10 actual values:", y_test.values[:10])
```

#### 6.2 Mean Absolute Error (MAE)
Average absolute difference between actual and predicted values.
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")  # Lower is better
```

#### 6.3 Mean Squared Error (MSE) & Root Mean Squared Error (RMSE)
```python
from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")  # Lower is better
```

#### 6.4 R-squared (R²) Score
Measures how well predictions fit actual values (0 to 1, closer to 1 is better).
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")  # Higher is better (closer to 1 is best)
```

#### 6.5 Mean Absolute Percentage Error (MAPE)
```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.2f}%")
```

---

## Step 7: Visualization

### What is it?
Creating graphs to understand model performance and data relationships.

### Visualizations:

#### 7.1 Actual vs Predicted
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Selling Price')
plt.grid(True)
plt.show()
```

#### 7.2 Residuals (Prediction Errors)
```python
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()
```

#### 7.3 Distribution of Residuals
```python
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
```

#### 7.4 Feature Importance (For Tree-based Models)
```python
# Only for Random Forest or similar models
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

---

## Step 8: Hyperparameter Tuning (Optional)

### What is it?
Fine-tuning model parameters to improve performance.

### Method: Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10]
}

# Create GridSearchCV
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

---

## Step 9: Making Predictions on New Data

### What is it?
Using your trained model to predict selling prices for new cars.

### Code:
```python
# Create a new car's data (must have same features as training data)
new_car = pd.DataFrame({
    'Year': [2020],
    'Present_Price': [10.5],
    'Kms_Driven': [50000],
    'Fuel_Type': [1],  # (encoded value)
    'Seller_Type': [0],  # (encoded value)
    'Transmission': [0],  # (encoded value)
    'Owner': [0]
})

# Scale using same scaler
new_car_scaled = scaler.transform(new_car)

# Make prediction
predicted_price = lr_model.predict(new_car_scaled)
print(f"Predicted Selling Price: {predicted_price[0]:.2f}")
```

---

## Step 10: Model Comparison & Selection

### What is it?
Comparing multiple models and choosing the best one.

### Code:
```python
models = {
    'Linear Regression': lr_model,
    'Ridge': ridge_model,
    'Lasso': lasso_model,
    'Random Forest': rf_model
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled) if name != 'Random Forest' else model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}

# Display comparison
results_df = pd.DataFrame(results).T
print(results_df)
```

---

## Summary

### Complete Workflow:
1. **Load & Explore** → Understand your data
2. **Preprocess** → Clean and encode data
3. **Split** → Divide into train/test sets
4. **Scale** → Normalize features
5. **Train** → Fit model on training data
6. **Evaluate** → Assess performance
7. **Visualize** → Create graphs
8. **Tune** → Improve hyperparameters
9. **Predict** → Make predictions on new data
10. **Compare** → Choose best model

---

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Features (X)** | Input variables used to make predictions |
| **Target (y)** | Variable you want to predict (Selling_Price) |
| **Training Set** | Data used to train the model (80%) |
| **Test Set** | Data used to evaluate the model (20%) |
| **Scaling** | Normalizing features to similar ranges |
| **Overfitting** | Model memorizes training data, poor on new data |
| **Underfitting** | Model too simple, poor on both train and test data |
| **R² Score** | Proportion of variance explained by model (0-1) |
| **RMSE** | Average magnitude of prediction errors |
| **MAE** | Average absolute prediction error |

---

## Tips for Success

✅ **Do:**
- Start with simpler models before complex ones
- Always scale features for linear models
- Use cross-validation for better evaluation
- Check for data leakage
- Visualize your results

❌ **Don't:**
- Train and test on same data
- Forget to scale features
- Ignore outliers without investigation
- Use test data for any tuning
- Ignore evaluation metrics

---

## Useful Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

---

**Remember:** The goal is not just to build a model, but to build a model that generalizes well to unseen data!

