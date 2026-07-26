def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)