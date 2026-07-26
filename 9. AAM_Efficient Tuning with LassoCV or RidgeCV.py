from sklearn.linear_model import LassoCV # Create a range of αs (lambda values)
αs = np.logspace(-4, 2, 50) # LassoCV automatically performs K-Fold Cross-Validation
lasso_cv = LassoCV(αs=αs, cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)
print(fOptimal lambda found: lasso_cv.α_)