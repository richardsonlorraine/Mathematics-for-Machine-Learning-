import numpy as np
def feed_forward_layer(X, W1, b1, W2, b2): # Step 1: Linear Expansion
    # X shape: (batch_size, seq_len, d_model)
    # W1 shape: (d_model, d_ff)
    hidden_pre = np.matmul(X, W1) + b1 # Step 2: Activation (ReLU)
    hidden_post = np.maximum(0, hidden_pre) # Step 3: Linear Compression
    # W2 shape: (d_ff, d_model)
    output = np.matmul(hidden_post, W2) + b2    
    return output