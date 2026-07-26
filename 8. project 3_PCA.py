import numpy as np
def pca_from_scratch(X, n_components): # 1. Standardize the data (Mean = 0, Std = 1)
    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered / np.std(X_centered, axis=0) # 2. Compute the Covariance Matrix # Covariance is the dot product of the transposed matrix and itself, divided by (n-1)
    cov_matrix = np.cov(X_scaled, rowvar=False) # 3. Eigen-decomposition # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # 4. Sort components # eight returns them in ascending order, so we reverse them
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index] # Select the top 'n' components
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components] # 5. Transform the data (Projection)
    X_reduced = np.dot(X_scaled, eigenvector_subset)
    return X_reduced, sorted_eigenvalues