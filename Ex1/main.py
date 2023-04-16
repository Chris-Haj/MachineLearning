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
