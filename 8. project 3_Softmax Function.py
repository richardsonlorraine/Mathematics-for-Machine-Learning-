def softmax(z):
    shifted_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
def softmax_derivative(s): # s is the softmax output vector. 
    # Reshape to a column vector for matrix multiplication
    s_col = s.reshape(-1, 1) # Jacobian matrix calculation
    return np.diagflat(s) - np.dot(s_col, s_col.T)