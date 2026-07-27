import numpy as np
def bayesian_update(mu_prior, sigma_prior, X, y, α=1.0, beta=1.0):
    mu_prior: Mean of weights
    sigma_prior: Covariance of weights
    X: Input features
    y: Observed targets
    α: Precision of prior
    beta: Precision of noise (1/variance) # Precision matrix (inverse of covariance)
    sigma_prior_inv = np.linalg.pinv(sigma_prior) # 1. Posterior Covariance: Sigma_post = (Sigma_prior-1 + beta * X.T @ X)-1
    sigma_post = np.linalg.pinv(sigma_prior_inv + beta * X.T @ X) # 2. Posterior Mean: mu_post = Sigma_post @ (Sigma_prior^-1 @ mu_prior + beta * X.T @ y)
    mu_post = sigma_post @ (sigma_prior_inv @ mu_prior + beta * X.T @ y)
    return mu_post, sigma_post # --- Usage Example ---
# Assume 2 features (intercept + x)
n_features = 2
mu_prior = np.zeros(n_features)
sigma_prior = np.eye(n_features) * 0.5 # Generate dummy data
X = np.array([[1, 0.5], [1, 1.0], [1, 1.5]]) # Design matrix (with intercept)
y = np.array([1.5, 2.0, 2.5])                # Observed targets
mu_post, sigma_post = bayesian_update(mu_prior, sigma_prior, X, y)
print(Updated Weight Mean:\n, mu_post)
print(Updated Weight Covariance:\n, sigma_post)
