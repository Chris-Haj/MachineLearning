import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""File given with data related to cancer patients, where the last column is the value of y we want to predict
(patient's life expectancy). Read the data and input it into matrix X and vector y.

a) Normalize the data (check that after normalization, the mean is indeed 0 and the standard deviation is 1).
b) Add a column of ones to matrix X.
c) Write a function that receives a vector x and returns (in the case of linear regression).
d) Write a function that receives a vector and the X and y matrices and returns the value of.
e) Write a function that receives a vector and the X and y matrices and returns the value of.
f) Run the Gradient Descent algorithm with several values of (for example, 1, 0.1, 0.01, 0.001)
and plot the graph showing the value of as a function of time steps using matplotlib.
g) Run the same code with mini-batches equal to the previous section. What are your conclusions from the run?
h) We learned in class 3 algorithms Momentum, Adagrad, Adam.
Choose at least one of them and run it. Is there faster convergence? Display a suitable graph. Write conclusions.

The assignment requires submission in pairs, and to add to the assignment (as a separate file or within the Python notebook)
 conclusions from the runs related to comparison between runs, convergence, etc."""


def createDataSet():
    data = pd.read_csv('cancer_data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return data, x, y


def normalizeData(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def addColumnOfOnes(x):
    x = np.insert(x, 0, 1, axis=1)
    return x


def hypothesis(x, theta):
    return np.dot(x, theta)


def costFunction(x, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum(np.square(hypothesis(x, theta) - y))


def gradientDescent(x, y, theta, alpha, iterations):
    m = len(y)
    costHistory = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - (alpha / m) * np.dot(x.T, (hypothesis(x, theta) - y))
        costHistory[i] = costFunction(x, y, theta)
    return theta, costHistory


def miniBatchGradientDescent(x, y, theta, alpha, iterations, batchSize):
    m = len(y)
    costHistory = np.zeros(iterations)
    for i in range(iterations):
        for j in range(0, m, batchSize):
            theta = theta - (alpha / m) * np.dot(x[j:j + batchSize].T,
                                                 (hypothesis(x[j:j + batchSize], theta) - y[j:j + batchSize]))
        costHistory[i] = costFunction(x, y, theta)
    return theta, costHistory


def adagradGradientDescent(x, y, theta, alpha, iterations, batchSize):
    m = len(y)
    costHistory = np.zeros(iterations)
    G = np.zeros(theta.shape)
    for i in range(iterations):
        for j in range(0, m, batchSize):
            G += np.dot(x[j:j + batchSize].T, (hypothesis(x[j:j + batchSize], theta) - y[j:j + batchSize])) ** 2
            theta = theta - (alpha / np.sqrt(G)) * np.dot(x[j:j + batchSize].T,
                                                          (hypothesis(x[j:j + batchSize], theta) - y[j:j + batchSize]))
        costHistory[i] = costFunction(x, y, theta)
    return theta, costHistory


def adamGradientDescent(x, y, theta, alpha, iterations, batchSize, beta1, beta2, epsilon):
    m = len(y)
    costHistory = np.zeros(iterations)
    v = np.zeros(theta.shape)
    s = np.zeros(theta.shape)
    for i in range(iterations):
        for j in range(0, m, batchSize):
            v = beta1 * v + (1 - beta1) * np.dot(x[j:j + batchSize].T,
                                                 (hypothesis(x[j:j + batchSize], theta) - y[j:j + batchSize]))
            s = beta2 * s + (1 - beta2) * np.dot(x[j:j + batchSize].T,
                                                 (hypothesis(x[j:j + batchSize], theta) - y[j:j + batchSize])) ** 2
            vCorrected = v / (1 - beta1 ** (i + 1))
            sCorrected = s / (1 - beta2 ** (i + 1))
            theta = theta - (alpha / np.sqrt(sCorrected + epsilon)) * vCorrected
        costHistory[i] = costFunction(x, y, theta)
    return theta, costHistory


if __name__ == '__main__':
    test = [5,3,1,2]


    pass


    data, x, y = createDataSet()
    x = addColumnOfOnes(normalizeData(x))
    theta = np.zeros(x.shape[1])
    alphas = [0.1, 0.01, 0.001]
    iterations = 1000
    batchSize = 32

    for i,alpha in enumerate(alphas):
        theta, costHistory = gradientDescent(x, y, theta, alpha, iterations)
        print("Theta: ", theta)
        print("Cost: ", costHistory[-1])
        plt.plot(range(iterations), costHistory)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title(f"Cost vs Iterations Gradient Descent {alpha,i}")
        plt.show()
    for i, alpha in enumerate(alphas):
        theta, costHistory = miniBatchGradientDescent(x, y, theta, alpha, iterations, batchSize)
        print("Theta: ", theta)
        print("Cost: ", costHistory[-1])
        plt.plot(range(iterations), costHistory)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title(f"Cost vs Iterations Minibatch {alpha,i}")
        plt.show()

    for i, alpha in enumerate(alphas):
        theta, costHistory = adagradGradientDescent(x, y, theta, alpha, iterations, batchSize)
        print("Theta: ", theta)
        print("Cost: ", costHistory[-1])
        plt.plot(range(iterations), costHistory)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title(f"Cost vs Iterations Adagrad {alpha,i}")
        plt.show()
