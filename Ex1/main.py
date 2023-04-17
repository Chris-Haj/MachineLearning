import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""File given with data related to cancer patients, where the last column is the value of y we want to predict
(patient's life expectancy). Read the data and input it into matrix X and vector y.
a) Normalize the data (check that after normalization, the mean is indeed 0 and the standard deviation is 1).
b) Add a column of ones to matrix X.
c) Write a function that receives a vector x and returns (in the case of linear regression).
d) Write a function that receives a vector and the X and y matrices and returns the value of.
e) Write a function that receives a vector and the X and y matrices and returns the value of.
f) Run the Gradient Descent algorithm with several values of (for example, 1, 0.1, 0.01, 0.001)
and plot the graph showing the value of as a function of time steps.
g) Run the same code with mini-batches equal to the previous section. What are your conclusions from the run?
h) We learned in class 3 algorithms Momentum, Adagrad, Adam.
Choose at least one of them and run it. Is there faster convergence? Display a suitable graph. Write conclusions.

The assignment requires submission in pairs, and to add to the assignment (as a separate file or within the Python notebook)
 conclusions from the runs related to comparison between runs, convergence, etc."""


# Read the data and input it into matrix X and vector y.
def createDataSet():
    data = pd.read_csv('cancer_data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return x, y


# a) Normalize the data (check that after normalization, the mean is indeed 0 and the standard deviation is 1).
def normalizeData(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


# b) Add a column of ones to matrix X.
def addOnesColumn(x):
    x = np.insert(x, 0, 1, axis=1)
    return x


# c) Write a function that receives a vector x and returns (in the case of linear regression).
def hypothesis(x, theta):
    return np.dot(x, theta)


# d) Write a function that receives a vector and the X and y matrices and returns the value of.
def costFunction(x, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum(np.square(hypothesis(x, theta) - y))


# e) Write a function that receives a vector and the X and y matrices and returns the value of.
def costFunctionDerivative(x, y, theta):
    m = len(y)
    return (1 / m) * np.dot(x.T, (hypothesis(x, theta) - y))


# f) Run the Gradient Descent algorithm with several values of (for example, 1, 0.1, 0.01, 0.001) and plot the graph showing the value of as a function of time steps.
def gradientDescent(x, y, theta, alpha, iterations):
    m = len(y)
    costHistory = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - (alpha * costFunctionDerivative(x, y, theta))
        costHistory[i] = costFunction(x, y, theta)
    return theta, costHistory



# g) Run the same code with mini-batches equal to the previous section. What are your conclusions from the run?
# h) We learned in class 3 algorithms Momentum, Adagrad, Adam.

if __name__ == '__main__':

