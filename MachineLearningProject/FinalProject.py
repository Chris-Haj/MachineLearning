import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd

# Step 1: Read the data
data = pd.read_csv('cancer_data.csv')
X = data.iloc[:, :-1].values  # assuming the last column is the target variable
y = data.iloc[:, -1].values

# Step 2: Normalize the data
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std
y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std

# Step 3: Add a column of ones to the data matrix for the intercept
X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))


# Step 4: Write the cost function for linear regression
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost


# Step 6: Gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []  # to record the cost in every iteration

    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        theta -= (alpha / m) * error
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


# Step 6: Visualize the convergence
def plot_convergence(J_history, title='Convergence of Gradient Descent'):
    plt.plot(J_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(title)
    plt.show()


# Initializations
theta = np.zeros(X_normalized.shape[1])
alpha = 0.005
iterations = 1000

# Run Gradient Descent
theta, J_history = gradient_descent(X_normalized, y_normalized, theta, alpha, iterations)

# Plotting the convergence of the cost function
plot_convergence(J_history)


# Step 7: Mini-batch gradient descent function
def mini_batch_gradient_descent(X, y, theta, alpha, batch_size):
    m = len(y)
    J_history = []
    n_batches = int(m / batch_size)


    for i in range(n_batches):
        indices = np.random.permutation(m)
        print(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for batch in range(batch_size):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch.dot(theta)
            error = np.dot(X_batch.transpose(), (predictions - y_batch))
            theta -= (alpha / batch_size) * error

            # Compute cost for the whole dataset to monitor the progress
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


# Settings for mini-batch gradient descent
batch_size = 16  # Adjust based on your dataset and computational resources
theta_mini_batch, J_history_mini_batch = mini_batch_gradient_descent(X_normalized, y_normalized, theta, alpha, batch_size)

# Plot convergence graph for mini-batch gradient descent
plot_convergence(J_history_mini_batch, title='Convergence of Mini-Batch Gradient Descent')


# Step 8: Dimensionality reduction with SVD
def reduce_dimensions(X, n_components):
    U, s, Vt = svd(X, full_matrices=False)
    Vt_reduced = Vt[:n_components, :]
    X_reduced = X.dot(Vt_reduced.T)
    return X_reduced



