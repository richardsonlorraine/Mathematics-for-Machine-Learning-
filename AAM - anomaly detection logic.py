# Simple anomaly detection logic
residuals = y_test - y_pred
std_res = np.std(residuals) # Flag anomalies where error is 3 standard deviations
anomalies = np.where(np.abs(residuals) 3 * std_res)[0]