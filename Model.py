import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load California housing data
data = fetch_california_housing(as_frame=True)
df = data.frame

# Split features and target
X = df.drop('MedHouseVal', axis=1)  # Drop target column
y = df['MedHouseVal']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grids for hyperparameter tuning
param_dist_rf = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

param_dist_gb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 10, 15],
    'subsample': [0.8, 1.0]
}

param_dist_xgb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 10, 15],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Model training function
def train_model(model, param_dist, name):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 {name} Results:")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE: {mae:.4f}")
    print(f" R²: {r2:.4f}")
    return best_model, rmse

# Train and evaluate models
rf_model, rf_rmse = train_model(RandomForestRegressor(random_state=42), param_dist_rf, "Random Forest")
gb_model, gb_rmse = train_model(GradientBoostingRegressor(random_state=42), param_dist_gb, "Gradient Boosting")
xgb_model, xgb_rmse = train_model(XGBRegressor(objective='reg:squarederror', random_state=42), param_dist_xgb, "XGBoost")

# Select the best model
best_model = min([(rf_model, rf_rmse), (gb_model, gb_rmse), (xgb_model, xgb_rmse)], key=lambda x: x[1])[0]

# Save the best model
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("✅ Best model saved as best_model.pkl")