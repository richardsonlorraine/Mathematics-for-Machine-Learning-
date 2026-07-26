import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(a): # a is the output of the sigmoid function: a = sigmoid(z)
    return a * (1 - a)