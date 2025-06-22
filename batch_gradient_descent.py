import numpy as np
import matplotlib.pyplot as plt

# Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate, n_iterations):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]

    theta = np.zeros(X_b.shape[1])
    cost_history = []

    # Iterate for n_iterations, basically until the cost is minimized/converged
    for _ in range(n_iterations):
        # Calculate predictions
        predictions = X_b.dot(theta)

        # Calculate error
        errors = predictions - y

        # Calculate gradient: 1/m * X_b_transpose * errors
        gradient = (1/m) * X_b.T.dot(errors)

        # Update parameters
        theta = theta - learning_rate * gradient

        # Calculate and store the cost (Mean Squared Error)
        cost = .5 * (1/m) * np.sum(errors**2)
        cost_history.append(cost)

    return theta, cost_history

# Example:
if __name__ == "__main__":
    # Generate some synthetic data for linear regression
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1)
    y = 2 * X + 3 + np.random.randn(m, 1)

    learning_rate = 0.01
    n_iterations = 5000

    # Run Batch Gradient Descent
    theta_optimized, cost_values = batch_gradient_descent(X, y.flatten(), learning_rate, n_iterations)

    print("Optimized Parameters (theta):", theta_optimized)
    print("Final Cost:", cost_values[-1])

    # View convergence
    plt.plot(range(n_iterations), cost_values)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.show()