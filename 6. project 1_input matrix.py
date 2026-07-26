import numpy as np # Define input matrix (e.g., 2 samples, 3 features)
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Define weight matrix (e.g., 3 features, 2 output neurons)
W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) # Matrix multiplication using the @ operator
output = X @ W 
print(output)