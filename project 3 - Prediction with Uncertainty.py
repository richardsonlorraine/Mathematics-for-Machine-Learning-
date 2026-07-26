# Conceptual implementation
def predict_bayesian(x, posterior_mean, posterior_cov): # Mean prediction
    y_pred = np.dot(x, posterior_mean) # Variance (Uncertainty)
    # The variance comes from both the model uncertainty and the data noise
    y_std = np.sqrt(np.dot(np.dot(x, posterior_cov), x.T))
    return y_pred, y_std