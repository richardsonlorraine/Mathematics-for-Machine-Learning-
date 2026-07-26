def relu(z):
    return np.maximum(0, z)
def relu_derivative(z):
    return np.where(z > 0, 1, 0)