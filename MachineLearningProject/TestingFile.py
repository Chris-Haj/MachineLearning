import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd

# Step 1: Read the data


data = pd.read_csv('cancer_data.csv', header=None)

X = data.iloc[:, :-1]  # Data set without life expectancy
y = data.iloc[:, -1]  # Data to predict (Theta will be compared to this)

"""
Calculate mean and std of data to normalize and standarise the data 
"""


def normalize(matrix):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return mean, std, (matrix - mean) / std


xMean, xSTD, normalX = normalize(X)
yMean, ySTD, normalY = normalize(y)

"""
Print the normalized matricies of X and Y to show their mean is 0
and their std is 1
"""

"""
Take normalized matrix as input and print their mean and std if they are 0 and 1
"""
def showMeanSTD(normalizedMatrix):
    print(f"Mean: {(np.mean(normalizedMatrix, axis=0)).round(2)}")
    print(f"STD: {np.std(normalizedMatrix, axis=0).round(2)}")



"""
Add column of ones to the x matrix in its first index 
"""
normalX = np.hstack((np.ones((normalX.shape[0], 1)), normalX))

"""
Take in x and theta and use them to calculate the prediction
hx(theta)
"""
def predict(X, theta):
    return np.dot(X, theta)


"""
Take in X,y,theta and use the predict function to calculate
the hx(theta) aka the predicitions and then calculate the Error
"""
def costFunction(X, y, theta):
    m = len(y)
    predictions = predict(X,theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient(X, y, theta):
    m = len(y)  # Number of training examples
    predictions = predict(X, theta)
    errors = predictions - y
    grad = 1 / m * np.dot(X.T, errors)
    return grad


# Gradient descent algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = []  # To store the cost in each iteration
    for i in range(iterations):
        theta -= alpha * gradient(X, y, theta)
        cost = costFunction(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

alpha = [0.001, 0.01, 0.1, 0.5]
iterations = 1000

def gradientDescentPlotting(alpha,iterations):
    for i in alpha:
        theta = np.zeros(normalX.shape[1])
        theta, cost_history = gradient_descent(normalX, normalY, theta, i, iterations)

        # Plotting the cost history to show convergence
        plt.figure(figsize=(8, 6))
        plt.plot(cost_history, label=f'Cost J with alpha = {i}')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Convergence of Gradient Descent')
        plt.legend()
        plt.show()

# def miniBatchGradient(X, y, theta, batchSize):
#     m = len(y)
#     indices = np.random.permutation(m)
#     X_shuffled = X[indices]
#     y_shuffled = y[indices]
#     gradients = []
#
#     for i in range(0, m, batchSize):
#         X_i = X_shuffled[i:i+batchSize]
#         y_i = y_shuffled[i:i+batchSize]
#         predictions = predict(X_i, theta)
#         errors = predictions - y_i
#         grad = 1 / len(X_i) * np.dot(X_i.T, errors)
#         gradients.append(grad)
#
#     return gradients


def miniBatchGradientDescent(X, y, theta, alpha, iterations, batchSize):
    m = len(y)
    cost_history = []

    for it in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batchSize):
            X_i = X_shuffled[i:i + batchSize]
            y_i = y_shuffled[i:i + batchSize]
            predictions = predict(X_i, theta)
            errors = predictions - y_i
            grad = 1 / len(X_i) * np.dot(X_i.T, errors)
            theta -= alpha * grad

        cost = costFunction(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

theta = np.zeros(normalX.shape[1])

# Run mini-batch gradient descent
def miniBatchGradientDescentPlotting(alpha,iterations,batchSize):
    for i in alpha:
        theta = np.zeros(normalX.shape[1])
        theta, cost_history = miniBatchGradientDescent(normalX, normalY, theta, i, iterations,batchSize)


        plt.figure(figsize=(8, 6))
        plt.plot(cost_history, label=f'Cost J with alpha = {i}')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Convergence of Mini-Batch Gradient Descent')
        plt.legend()
        plt.show()

alpha = [0.0001, 0.001, 0.01]  # Reduced learning rates


miniBatchGradientDescentPlotting(alpha,500,10)