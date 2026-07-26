# Calculate total variance
total_var = np.sum(sorted_eigenvalues)
# Calculate ratio per component
explained_variance_ratio = sorted_eigenvalues / total_var
print(fVariance explained by first 2 components: np.sum(explained_variance_ratio[:2]):.2%)