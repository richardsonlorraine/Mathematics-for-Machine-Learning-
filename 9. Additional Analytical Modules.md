Additional Analytical Modules

* Predictive Maintenance via Residual Analysis: Uses linear regression residuals and standard deviation thresholds to detect mechanical anomalies in sensor data.
* Regularization and Cross-Validation: Explores Lasso (L1) and Ridge (L2) penalties alongside K-Fold cross-validation to manage the bias-variance tradeoff and perform automated feature selection.

Anomaly Detection via Residuals: In industrial IoT, you often have machines that run consistently. Linear regression can model the normal behavior of a machine, and you can flag deviations as potential failures.

The Goal: Build a model that predicts a machine's temperature based on its speed, then use the Residuals (the errors) to trigger an alert if a data point falls outside a statistical threshold.

The Math:

* Linear Model: y = β0 + β1 x + ε
* Residuals: ei = yi - yi (the vertical distance between the actual data and your regression line).

Standardization: Calculate the Z-score of your residuals to determine if a point is a statistically significant outlier.

Implementation Roadmap

1. Modeling Normalacy: Train a simple Linear Regression model on a healthy dataset. This captures the expected relationship between speed and temperature.

2. Calculating Residuals: After training, calculate the difference between the actual observed temperature and your model's predicted temperature for every new data point.

3. Defining the Threshold: Calculate the Standard Deviation of your residuals (Σres). 
If any new residual ei 3 • Σres, the machine is behaving in a way the model didn't expect.

# Simple anomaly detection logic

residuals = y_test - y_pred

std_res = np.std(residuals) # Flag anomalies where error is 3 standard deviations

anomalies = np.where(np.abs(residuals) 3 * std_res)[0]

1. Visualize the Confidence: Use plt.fill_between to plot the 95% confidence interval around your regression line. Any data point falling outside this shaded region is your anomaly.

2. Explain the Why: In your portfolio, explain that this is a Unsupervised-adjacent approach to predictive maintenance. You are using the error of your regression as a proxy for machine health.

3. Real-world Context: Mention that this is how real-world sensors detect hardware degradation—by identifying when normal operating ranges are no longer being followed.

I implemented a predictive maintenance model using Linear Regression to detect anomalies in mechanical sensor data. By analyzing the residuals of the model, I established a dynamic threshold for identifying abnormal behavior. This project demonstrates how simple statistical models can be repurposed for robust, real-world monitoring applications.

Regularization is the mathematical remedy for Overfitting. When you have many features, a standard linear regression model might memorize the noise in your data rather than the signal.

Lasso (L1) and Ridge (L2) regularization add a penalty term to your loss function, forcing the model to be simpler and more robust.

The Mathematical Penalty: In standard Linear Regression, we minimize the Sum of Squared Errors (SSE). Regularization adds a penalty term:

* Ridge (L2) Penalty: Adds the sum of squared weights: λΣwi2
* Lasso (L1) Penalty: Adds the sum of absolute weights: λΣ|wi|

The constant λ (lambda) controls the strength of the penalty. As λ increases, the model prioritizes smaller weights to reduce the penalty, which reduces overfitting.

Key Difference for Feature Selection

* Ridge shrinks weights toward zero but never makes them exactly zero. It’s best for preventing overfitting when many features are important.
* Lasso can force weights to exactly zero. This acts as an automated feature selection tool, effectively removing irrelevant features from your model.

Implementation: Using sklearn: While you can write this with numpy using the Normal Equation, scikit-learn is the industry standard for these specific algorithms because of its highly optimized solvers.

from sklearn.linear_model import Ridge, Lasso

from sklearn.preprocessing import StandardScaler # 1. Scaling is CRITICAL

# Regularization penalizes the magnitude of weights, so features must be on the same scale.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X) # 2. Ridge Regression (Keeps all features, shrinks weights)

ridge = Ridge(α=1.0)

ridge.fit(X_scaled, y) # 3. Lasso Regression (Performs feature selection by setting coefficients to 0)

lasso = Lasso(α=0.1)

lasso.fit(X_scaled, y) # Inspect features

print(Lasso coefficients:, lasso.coef_) # Any coefficient that is 0 was 'dropped' by Lasso

Create a Feature Importance chart to show your model in action:

 1. Generate a noisy dataset: Create 20 features, but only 3 of them actually correlate with your target.

 2. Train Lasso: Set a range of α values.

 3. Visualize: Plot how the coefficients change as α increases. You will see 17 of your feature lines drop to zero, leaving only the 3 important ones.

To improve model generalizability and perform automated feature selection, I implemented Lasso and Ridge regression. By tuning the regularization parameter (λ), I demonstrated the ability to eliminate redundant input variables, resulting in a more interpretable model that is significantly more resistant to noise than standard Ordinary Least Squares (OLS) regression.

Cross-Validation is the standard technique for finding the optimal hyperparameter λ (the strength of your regularization). Because we cannot know the perfect λ beforehand, we must test a range of values to see which one produces the best balance between bias and variance.

The Intuition: Trial and Error with Fairness: If you test your model on the same data you used to train it, the model will just memorize the data and pick a λ that overfits. Cross-validation solves this by splitting your training data into K parts (folds).

1. You train the model on K-1 folds.

2. You validate the model on the remaining 1 fold (the unseen data).

3. You repeat this K times, rotating which fold is the validation set.

4. You average the performance (the error) to get a score for that specific λ.

The Mathematical Optimization Process: To find the optimal λ, you are essentially performing a grid search over a logarithmic scale:

 1. Define a Grid: Choose a set of candidates for λ (e.g., 10-4, 10-3, ..., 102).

 2. Evaluate: For each λ, perform K-Fold Cross-Validation.

* The Cross-Validation score CV(λ) is the average of the error across all K folds:

3. Minimize: Select the λ that results in the lowest CV(λ) score.

Implementation: Efficient Tuning with LassoCV or RidgeCV: In practice, you rarely need to write the K-Fold loop manually. Scikit-learn provides CV versions of these models that handle the optimization internally and efficiently.

from sklearn.linear_model import LassoCV # Create a range of αs (lambda values)

αs = np.logspace(-4, 2, 50) # LassoCV automatically performs K-Fold Cross-Validation

lasso_cv = LassoCV(αs=αs, cv=5, random_state=42)

lasso_cv.fit(X_scaled, y)

print(fOptimal lambda found: lasso_cv.α_)

* Bias-Variance Tradeoff: State that small λ values lead to high variance (overfitting), while large λ values lead to high bias (underfitting). Cross-validation is the search for the sweet spot in the middle.
* The Validation Curve: Generate a plot showing λ on the X-axis and Mean Squared Error on the Y-axis. The point where the curve hits its minimum is your optimal λ. This visual proof confirms your model is optimized, not just guessed.

Note: If you want to make your project stand out even more, mention Nested Cross-Validation. This is used when you also need to tune other hyperparameters alongside λ, preventing data leakage where the model implicitly learns about the test set during the tuning process.

Lessons Learned

* Regularisation demonstrated that improving model performance is often achieved by controlling model complexity rather than increasing it.
* Ridge and Lasso regression illustrated different strategies for reducing overfitting while improving model generalisation.
* Cross-validation reinforced the importance of evaluating models on unseen data rather than relying solely on training performance.
* Selecting hyperparameters systematically produces more reliable models than manual experimentation.
* These techniques highlighted that successful machine learning depends on balancing mathematical theory with practical engineering judgement.