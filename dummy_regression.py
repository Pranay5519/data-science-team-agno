import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Create a dummy dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, noise=0.5, random_state=42)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# 3. Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions and evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R-squared score on the test set: {r2:.4f}")

# 5. Print feature importances
feature_importances = model.feature_importances_
print(f"Feature Importances: {feature_importances}")