"""
House Price Prediction Model Development
Student: Wesley Iluobe
Matric Number: 23CH034193

This script builds and trains a Random Forest Regressor model
to predict house prices based on 6 selected features from the
Kaggle "House Prices: Advanced Regression Techniques" dataset.

Features used:
1. OverallQual - Overall material and finish quality (1-10)
2. GrLivArea - Above grade living area (square feet)
3. TotalBsmtSF - Total basement square footage
4. GarageCars - Garage capacity (number of cars)
5. FullBath - Number of full bathrooms
6. YearBuilt - Original construction year
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD THE DATASET
# ============================================
print("=" * 60)
print("HOUSE PRICE PREDICTION MODEL DEVELOPMENT")
print("Student: Wesley Iluobe | Matric: 23CH034193")
print("=" * 60)

print("\n[1] Loading dataset...")
df = pd.read_csv('train.csv')
print(f"    Dataset loaded successfully!")
print(f"    Total samples: {len(df)}")
print(f"    Total features: {len(df.columns)}")

# ============================================
# 2. FEATURE SELECTION
# ============================================
print("\n[2] Selecting features...")
# Selected 6 features from the allowed 9
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
target = 'SalePrice'

print(f"    Selected features: {selected_features}")
print(f"    Target variable: {target}")

# Create feature matrix and target vector
X = df[selected_features].copy()
y = df[target].copy()

print(f"\n    Feature matrix shape: {X.shape}")
print(f"    Target vector shape: {y.shape}")

# ============================================
# 3. DATA PREPROCESSING
# ============================================
print("\n[3] Data Preprocessing...")

# 3a. Handling missing values
print("\n    3a. Checking for missing values...")
missing_before = X.isnull().sum()
print(f"        Missing values before handling:")
for col in selected_features:
    if missing_before[col] > 0:
        print(f"        - {col}: {missing_before[col]}")

# Fill missing values with median (for numerical features)
for col in selected_features:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"        Filled {col} missing values with median: {median_val}")

missing_after = X.isnull().sum().sum()
print(f"        Missing values after handling: {missing_after}")

# 3b. Feature scaling
print("\n    3b. Applying feature scaling (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

print(f"        Scaling complete!")
print(f"        Feature means (after scaling): ~0")
print(f"        Feature stds (after scaling): ~1")

# ============================================
# 4. SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================
print("\n[4] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"    Training set size: {len(X_train)}")
print(f"    Test set size: {len(X_test)}")

# ============================================
# 5. TRAIN THE MODEL (Random Forest Regressor)
# ============================================
print("\n[5] Training Random Forest Regressor model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("    Model trained successfully!")

# Feature importance
print("\n    Feature Importance:")
importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"    - {row['Feature']}: {row['Importance']:.4f}")

# ============================================
# 6. MODEL EVALUATION
# ============================================
print("\n[6] Evaluating model performance...")

# Predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n    Performance Metrics on Test Set:")
print(f"    --------------------------------")
print(f"    Mean Absolute Error (MAE):     ${mae:,.2f}")
print(f"    Mean Squared Error (MSE):      ${mse:,.2f}")
print(f"    Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"    R-squared (R²):                 {r2:.4f}")

# ============================================
# 7. SAVE THE MODEL
# ============================================
print("\n[7] Saving model and scaler to disk...")

# Save the model using joblib
model_path = 'model/house_price_model.pkl'
scaler_path = 'model/scaler.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"    Model saved to: {model_path}")
print(f"    Scaler saved to: {scaler_path}")

# ============================================
# 8. VERIFY MODEL CAN BE RELOADED
# ============================================
print("\n[8] Verifying model can be reloaded without retraining...")

# Load the saved model
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Make a test prediction
sample_data = pd.DataFrame({
    'OverallQual': [7],
    'GrLivArea': [1500],
    'TotalBsmtSF': [1000],
    'GarageCars': [2],
    'FullBath': [2],
    'YearBuilt': [2000]
})

sample_scaled = loaded_scaler.transform(sample_data)
sample_prediction = loaded_model.predict(sample_scaled)

print(f"    Test prediction with sample data:")
print(f"    Sample house features: {sample_data.iloc[0].to_dict()}")
print(f"    Predicted price: ${sample_prediction[0]:,.2f}")
print(f"\n    ✓ Model reloaded and working correctly!")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("MODEL DEVELOPMENT COMPLETE!")
print("=" * 60)
print(f"""
Summary:
- Algorithm: Random Forest Regressor
- Features Used: {len(selected_features)} features
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Model Performance (R²): {r2:.4f}
- Model saved using: Joblib

Files created:
- model/house_price_model.pkl (trained model)
- model/scaler.pkl (feature scaler)
""")
