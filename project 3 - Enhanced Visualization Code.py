import matplotlib.pyplot as plt
import numpy as np # ... (Assume explained_variance_ratio and cumulative_variance are calculated) ...
# Find the number of components needed for 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
plt.figure(figsize=(10, 6)) # Individual Variance
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, α=0.4, align='center', label='Individual Variance') # Cumulative Variance
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red', label='Cumulative Variance')
# Threshold line (90%)
plt.axhline(y=0.90, color='blue', linestyle='--', label='90% Threshold') # Highlight the crossing point
plt.axvline(x=n_components_90, color='green', linestyle=':', label=f'Components needed: n_components_90')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Scree Plot: Determining Optimal Components')
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--', α=0.3)
plt.tight_layout()
plt.show()