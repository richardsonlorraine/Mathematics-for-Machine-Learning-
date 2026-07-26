import numpy as np
def get_batches(X, y, batch_size): # Shuffle indices to randomize the order
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) # Yield batches
    for start_idx in range(0, X.shape[0], batch_size):
        batch_indices = indices[start_idx : start_idx + batch_size]
        yield X[batch_indices], y[batch_indices] # Usage in your training loop:
batch_size = 32
for epoch in range(100):
    for X_batch, y_batch in get_batches(X, y, batch_size): # Perform your forward pass and backpropagation on X_batch
        # Update weights using the gradients calculated from this batch