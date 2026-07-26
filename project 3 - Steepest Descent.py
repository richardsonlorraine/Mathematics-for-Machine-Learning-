def update_weights(weights, gradients, learning_rate): # This is the 'Steepest Descent' step
    return weights - (learning_rate * gradients)