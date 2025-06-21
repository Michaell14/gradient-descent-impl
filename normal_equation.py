import numpy as np
import matplotlib.pyplot as plt
# Theta = (X^T * X)^-1 * X^T * y

def normal_equation(X, y):
    # Args:
    #     X (numpy.ndarray): The design matrix. Must include a column of ones
    #                        for the intercept term if desired.
    #                        Shape: (m, n+1) where m is samples, n is features.
    #     y (numpy.ndarray): The target variable vector.
    #                        Shape: (m, 1) or (m,)

    # Returns:
    #     numpy.ndarray: The optimal parameter vector (theta).
    #                    Shape: (n+1, 1)
    X_T = np.transpose(X)
    X_T_X = np.matmul(X_T, X)
    X_T_X_inv = np.linalg.inv(X_T_X)

    X_T_y = np.matmul(X_T, y)
    res = np.matmul(X_T_X_inv, X_T_y)
    return res

def normal_equation_solve(X, y):
    """
    Calculates the optimal parameters (theta) for linear regression
    using the Normal Equation.

    Args:
        X (numpy.ndarray): The design matrix. Must include a column of ones
                           for the intercept term if desired.
                           Shape: (m, n+1) where m is samples, n is features.
        y (numpy.ndarray): The target variable vector.
                           Shape: (m, 1) or (m,)

    Returns:
        numpy.ndarray: The optimal parameter vector (theta).
                       Shape: (n+1, 1)
    """
    # Ensure y is a 2D column vector
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # Calculate X_transpose * X
    # X.T is the transpose of X
    X_T_X = X.T @ X  # Using @ for matrix multiplication (Python 3.5+)
    
    # Calculate the inverse of (X_transpose * X)
    # np.linalg.inv() calculates the inverse
    try:
        X_T_X_inv = np.linalg.inv(X_T_X)
    except np.linalg.LinAlgError:
        print("Warning: (X_T_X) is singular or near-singular. "
              "Consider using a different method like gradient descent "
              "or regularization if features are highly correlated or n > m.")
        # In a real application, you might want to fall back to a pseudo-inverse
        # np.linalg.pinv(X) for the entire X, or other robust methods.
        return None 

    # Calculate X_transpose * y
    X_T_y = X.T @ y

    # Calculate theta = (X_T_X_inv) * (X_T_y)
    theta = X_T_X_inv @ X_T_y

    return theta

# 1. Generate some sample data for linear regression
# Let's say we have one feature 'x' and a target 'y'
# y = 2 + 3*x + noise

# Generating data points
np.random.seed(42)
m = 100 # 100 data points
x = np.arange(100).reshape((-1,1))
ones = np.ones((100, 1))
X = np.hstack((ones, x))
delta = np.random.normal(0, 10, size = (100,1))
y = .4 * x + 3 + delta

res = normal_equation(X, y)
print(res)

# plt.plot(x, y, linestyle = "none", marker = ".")
# plt.show()