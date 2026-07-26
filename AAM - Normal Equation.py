from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler # 1. Scaling is CRITICAL
# Regularization penalizes the magnitude of weights, so features must be on the same scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 2. Ridge Regression (Keeps all features, shrinks weights)
ridge = Ridge(α=1.0)
ridge.fit(X_scaled, y) # 3. Lasso Regression (Performs feature selection by setting coefficients to 0)
lasso = Lasso(α=0.1)
lasso.fit(X_scaled, y) # Inspect features
print(Lasso coefficients:, lasso.coef_) # Any coefficient that is 0 was 'dropped' by Lasso