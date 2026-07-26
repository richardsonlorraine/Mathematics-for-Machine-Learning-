Project 1: From Scratch PCA for Image Compression

* Focuses on building Principal Component Analysis from the ground up using linear algebra foundations, including:  
	* covariance matrices and 
	* eigen-decomposition or 
	* Singular Value Decomposition (SVD).
* Utilizes matrix multiplication for data projection and dimensionality reduction, supported by scree plots to evaluate variance retention and compression performance.
From Scratch PCA for Image Compression: Instead of just using sklearn.decomposition.PCA, build it from the ground up using Linear Algebra.

The Mathematics:

Covariance Matrix: Construct the Covariance Matrix of your image data. Covariance analysis in machine learning quantifies the joint directional relationship between two or more variables. By organizing these relationships into a square N*N covariance matrix (where N is the number of features), ML algorithms can identify feature redundancies, perform dimensionality reduction, and model multidimensional data distributions.
How It Works: The covariance cov(x, y) between two features x and y calculates how much the two variables vary together from their respective means: Cov(X,Y)=Σni=1(xi-x̄)
* Positive value: Both variables tend to increase or decrease together.
* Negative value: As one variable increases, the other tends to decrease.
* Zero: No linear relationship exists between them.
In a covariance matrix, the main diagonal represents the variance of individual features, while the off-diagonal cells contain the pairwise covariances.

Key Applications in Machine Learning: 

* Dimensionality Reduction (PCA): Uses the covariance matrix to find directions (eigenvectors) that capture the highest variance, allowing algorithms to discard less informative features.
* Feature Selection: High covariance between two input features indicates redundancy. Removing one prevents multicollinearity, speeding up training and reducing overfitting.
* Gaussian Models: Algorithms like Linear Discriminant Analysis (LDA) or Gaussian Mixture Models use covariance matrices to map the shape and orientation of data clusters.
* Anomaly Detection: Helps establish a baseline normal distribution; data points deviating from this structure are flagged as anomalies.

Common Python Implementation: In Python, you can natively calculate and analyze covariance matrices using the NumPy or pandas libraries:

import numpy as np # Sample feature data: 

X = np.array([[10, 20, 30], [15, 25, 20], [30, 10, 50]]) # Calculate covariance matrix (rowvar=False treats columns as features)

cov_matrix = np.cov(X, rowvar=False)

print(cov_matrix)

Note: For advanced visualization, we often convert the covariance matrix into a correlation matrix by standardizing values between -1 and 1 to interpret relationship strength.

Use code with caution: For advanced visualization and modeling, we often convert the covariance matrix into a correlation matrix by standardizing the values between -1 and 1 to better interpret the strength of relationships.

Perform Eigen-decomposition (or SVD): Eigen-decomposition and Singular Value Decomposition (SVD) are foundational linear algebra techniques used to factorize matrices into simpler, interpretable components.

Key Differences & Mechanics

* Eigen-decomposition: Applies only to square matrices. It breaks a matrix A into eigenvectors (directions of variance) and eigenvalues (magnitude). Expressed as A = QλQ-1.
* SVD: Generalizes to any rectangular m*n matrix. Decomposes a matrix into A = UΣVT. SVD is generally preferred in machine learning for its superior numerical stability.
* Project Data onto Components (Matrix Multiplication): Matrix multiplication acts as a geometric transformation. It allows parallel processing of batches and performs scaling, rotation, and projection.
	* Constraint: The number of columns in A must equal the number of rows in B.
	* Mathematically: Cij = Σk=1K Aik Bkj

Machine Learning Use Cases

1. Dimensionality Reduction & PCA: The most common application is Principal Component Analysis (PCA). When reducing high-dimensional data, you calculate the covariance matrix of your dataset and extract its principal components. You can do this via:
* Eigen-decomposition: By applying it directly to the square covariance matrix.
* SVD: By applying it directly to the original, centered data matrix. SVD is generally preferred here because calculating the covariance matrix directly can cause a loss of numerical precision.

2. Recommender Systems (Matrix Factorization): Techniques popularized by the Netflix Prize use SVD to solve the collaborative filtering problem. By treating the user-item rating matrix as a highly sparse rectangular matrix, SVD allows models to discover latent features (e.g., hidden genre preferences) that predict unrated items.

3. Image Compression & Noise Filtering: SVD allows for low-rank approximations of high-resolution image matrices. By keeping only the largest singular values and setting smaller ones to zero, you can reconstruct an image using a fraction of the original data, successfully compressing the file size while preserving essential visual structure.

4. Natural Language Processing (Latent Semantic Analysis): In NLP, term-document matrices are extremely large and sparse. Applying SVD transforms this matrix into a lower-dimensional space, capturing the underlying semantic relationships between words and documents rather than just exact word counts.

Project the high-dimensional image data onto these components (Matrix Multiplication): Matrix multiplication is the foundational engine of machine learning. It acts as a geometric transformation that maps inputs to outputs. In neural networks, it applies layers of weights to data, allowing models to learn complex patterns, with acceleration typically handled by GPUs and TPUs.

Why It's Crucial:

* Parallel Processing: Instead of looping through thousands of data points one by one, matrix multiplication processes batches of data simultaneously.
* Geometric Transformations: Geometrically, it performs operations like scaling, rotation, and projection on data points.
* Layered Learning: Chaining these multiplications allows networks to learn the exact spatial mapping needed to categorize an image or translate a sentence.

Core Mechanics: For two matrices A and B to be multiplied, the number of columns in A must equal the number of rows in B.

* If matrix A is dimension M × K and matrix B is K × N, the resulting matrix will have dimensions M × N. The individual elements are computed by taking the dot product of rows from A and columns from B.
* Mathematically: Cij = Σk=1K Aik Bkj

Common Use Cases: 
* Linear & Logistic Regression: Used to apply coefficients to feature sets to compute predictions: ŷ = Xw + b
* Neural Networks / Deep Learning: Used to calculate hidden layer transformations: Z = W ⋅ X + bias
* Transformers & Attention: The foundational query, key, and value (Q, K, V) processes are executed via intensive matrix multiplications.

Python Implementation: In Python, this is almost universally handled using the NumPy library's dot product or the @ operator:

import numpy as np # Define input matrix (e.g., 2 samples, 3 features)

X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Define weight matrix (e.g., 3 features, 2 output neurons)

W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) # Matrix multiplication using the @ operator

output = X @ W 

print(output)

 * The Value: This proves you understand dimensionality reduction. By reconstructing the image using only the top k eigenvalues, you demonstrate how to trade off file size vs. visual quality.
 * Portfolio Highlight: I implemented PCA from scratch using NumPy to achieve a 10x reduction in image size while retaining 95% of structural variance.

Lessons Learned

* Implementing PCA from first principles reinforced how covariance matrices and eigenvectors determine the directions of maximum variance rather than treating PCA as a library function.
* Feature scaling and data centring proved essential, as even small preprocessing errors significantly affected the principal components.
* Visualising explained variance demonstrated the trade-off between compression and information retention, highlighting that reducing dimensionality always involves balancing efficiency against accuracy.
* Comparing the mathematical derivation with the Python implementation improved my understanding of numerical linear algebra and matrix operations.