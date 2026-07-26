import numpy as np # 1. Initialization
# Assume 3 input features, 4 hidden neurons, 1 output neuron
W1 = np.random.randn(3, 4) 
W2 = np.random.randn(4, 1)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x) # 2. The Training Loop
learning_rate = 0.1
for epoch in range(1000):  # --- Forward Pass ---
    hidden_layer_input = np.dot(X, W1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output = np.dot(hidden_layer_output, W2  # --- Error Calculation ---
    error = y - final_output  # --- Backward Pass (Backpropagation) ---
    # Calculate gradient for W2 using Chain Rule
    # d_loss/d_output * d_output/d_W2
    d_W2 = np.dot(hidden_layer_output.T, error)  # Calculate gradient for W1 using Chain Rule
    # Propagate error back to hidden layer, then multiply by derivative of hidden activation
    error_hidden = np.dot(error, W2.T)
    d_W1 = np.dot(X.T, error_hidden * sigmoid_derivative(hidden_layer_output)) # --- Weight Update (Steepest Descent) ---
    W2 += learning_rate * d_W2
    W1 += learning_rate * d_W1