import matplotlib.pyplot as plt
import numpy as np # Assuming 'sorted_eigenvalues' is the array from your PCA calculation
total_var = np.sum(sorted_eigenvalues)
explained_variance_ratio = sorted_eigenvalues / total_var
cumulative_variance = np.cumsum(explained_variance_ratio) # Plotting
plt.figure(figsize=(10, 6)) # Bar plot for individual variance
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, α=0.5, align='center', label='Individual explained variance') # Line plot for cumulative variance
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('scree_plot.png')