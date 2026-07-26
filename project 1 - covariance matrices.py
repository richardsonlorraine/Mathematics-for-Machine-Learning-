import numpy as np # Sample feature data: 
X = np.array([[10, 20, 30], [15, 25, 20], [30, 10, 50]]) # Calculate covariance matrix (rowvar=False treats columns as features)
cov_matrix = np.cov(X, rowvar=False)
print(cov_matrix)