import numpy as np

"""

Submission instructions:
    - only submit the `ex09_hw.py` file. Don't zip it!
    - any file other than `ex09_hw.py` will be ignored during the grading. 
        Failure to comply will result in a grade of 0.
    - Make sure to include your id number in the `get_id_number` function.
    - Do not change the function signatures.
    - Do not add any additional imports
    - if you want to add additional functions - you may do so
    - if you want to have test code, make sure it is inside a function or a `if __name__ == "__main__":` block. 
        Otherwise, it will be executed during the import and might fail the tests. In these case, the grade will 
        be 0 for the entire exercise!
    - if I try to import your file and it fails for any reason, you will get 0 points for the entire exercise.
    - if the function signature is wrong, you will get 0 points for that function.
    - if a function fails to run with my test code, you will get 0 points for that function.

"""


def get_id_number() -> str:
    """
    Return your ID number AS A STRING.
    You don't get points for this function. :-)
    If this function is not implemented, or if it does not return a string,
    you will get 0 points for the entire exercise!
    Keep in mind that printing is not returning. Don't use the print statement in this function,
    only the return statement.
    """
    return '207824772'


## Functions provided for your convenience


def widrow_hoff_least_squares(
        X, y, initial_guess=None, learning_rate=1e-2, epochs=1000
):
    """
    Implements the Widrow-Hoff least squares algorithm using gradient descent.

    Parameters:
    X : numpy.ndarray
        The input features, shape (n_samples, n_features).
    y : numpy.ndarray
        The target values, shape (n_samples,).
    learning_rate : float, optional
        The learning rate for gradient descent.
    epochs : int, optional
        The number of iterations to run the gradient descent algorithm.

    Returns:
    numpy.ndarray
        The optimized weights, shape (n_features,).
    """
    # Initialize weights with zeros
    if initial_guess is not None:
        weights = initial_guess
    else:
        weights = np.zeros(X.shape[1])  # weights are `params` from the previous example

    # Iterate over the number of epochs
    errors = []
    for _ in range(epochs):
        # Compute the predictions
        predictions = (
                X @ weights
        )  # The @ operator in Python is used for matrix multiplication.
        # Compute the error
        error = predictions - y
        # Compute the gradient
        gradient = X.T @ error / len(X)
        # Update the weights
        weights -= learning_rate * gradient
        errors.append(np.mean(error ** 2))
    return weights, errors


def generate_correlated_data(x: np.array, correlation: float) -> np.array:
    """
    Generate a vector similar in size to x, but with a given linear correlation to x.
    Args:
        x: Vector of numbers. Assume it is a 1D numpy array of floats with no NaNs or Infs.
        correlation: a float between -1 and 1. The desired correlation between x and the returned vector.

    Returns:
        numpy.ndarray: a vector of the same size as x, with the desired correlation to x.

    """
    ret = correlation * x + np.sqrt(1 - correlation ** 2) * np.random.normal(
        0, 1, size=len(x)
    )
    return ret


def pca_transform(data: np.ndarray) -> np.ndarray:
    """
    Perform PCA on the given data, retaining all the components.
    Args:
        data:
            a 2D numpy array with the data. Each row represents a sample. Each column represents a feature.
            We assume that the data does not contain any NaNs or Infs, only numbers
    Returns:
        a 2D numpy array with the transformed data. Each row represents a sample. Each column represents
        a transformed feature.

    """
    # Subtract the mean
    data -= np.mean(data, axis=0)
    # Calculate the covariance matrix
    cov = np.cov(data, rowvar=False)
    # Calculate the eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Transform the data
    return data @ eigvecs


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


## Questions for the student

class LeastSquaresLinearRegression:
    def __init__(
            self,
            learning_rate: float = 0.01,
            epochs: int = 1000,
            normalize_data_before_fit: bool = False,
    ):
        """
        Initialize the instance with the given parameters.

        Args:
            learning_rate: A float. The learning rate for the gradient descent algorithm.
            epochs: How many iterations to run the gradient descent algorithm.
            normalize_data_before_fit: If True, the data will be normalized before fitting the model.
                Otherwise, it will not be normalized.
                IMPORTANT: If the data is normalized prior to fitting, it is your responsibility to handle the
                test data and the predictions accordingly!
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.normalize_data_before_fit = normalize_data_before_fit
        self.is_fitted = False
        # Add anything else you need here
        # for example self.params = None ..
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the given data.
        Args:
            X: the input features, as a 2D numpy array. Each row represents a sample.
            Each column represents a feature.
                We assume that X does not contain any NaNs or Infs, only numbers
                We also assume that the intercept term is included in X.

            Note that it is your responsibility to add the intercept term to X if you want to use it.

            y: the dependent variable, as a 1D numpy array.
            We assume that y does not contain any NaNs or Infs, only numbers,
            and that it has the same number of rows as X.

        Returns: self
        """

        # Add your code here

        samples, features = X.shape
        self.weights = np.zeros(features)
        for _ in range(self.epochs):
            predictions = X.dot(self.weights)
            errors = predictions - y
            gradient = X.T.dot(errors) / samples
            self.weights -= self.learning_rate * gradient

        self.is_fitted = True
        # don't forget to return self!
        return self

    def get_fitted_parameters(self) -> np.ndarray:  # Function is finished
        """
        Return the fitted parameters of the model.

        I will run this method after calling the `fit` method using my data that I generated using known sets of
        parameters. I will test the returned parameters against the known parameters and if they are similar, you get
        points. If the parameters are not similar to the known parameters, but have the correct dimensions, you will
        one fifth of the points. Otherwise, you will get 0 points.

        Returns:
            if the model is fitted, return the fitted parameters as a 1D numpy array.
            otherwise, raise a RuntimeError.

         20 points
        """
        # add your code here
        if self.is_fitted:
            return self.weights
        raise RuntimeError('Model is not fitted')

    def predict(self, X: np.ndarray) -> np.ndarray:  # Function is finished
        """
        Return the model's predictions on the given data.

        I will run this method after calling the `fit` method using my data that I generated using known sets of
        parameters. I will compare the returned predictions to the known predictions and if they are similar, you get
        points. If the predictions are not similar to the known  but are close enough, you will get half the points.
        If the predictions are not close to the known predictions, but have the correct dimensions, you will get one
        fifth of the points. Otherwise, you will get 0 points.

        Args:
            X: the input features, as a 2D numpy array. Each row represents a sample. Each column represents a feature.
                We assume that X does not contain any NaNs or Infs, only numbers
                We also assume that the intercept term is included in X.

        Returns:
            A 1D numpy array with the model's predictions for each sample in X.

        20 points
        """
        if not self.is_fitted:
            raise RuntimeError("The model is not fitted")
        return X.dot(self.weights)


def explore_effect_of_correlation_on_least_squares(
        x1: np.ndarray, params: np.ndarray, pca_transform: bool = False
):
    """
    Explore the effect of correlation between the features on the performance of the least squares algorithm.
    This function accepts a vector of data, x1, and a vector of parameters, `params`. It generates a second vector,
    x2, that is correlated with x1. It then fits generates `y` using the known parameters and the generated x1 and x2.
    Next, it fits a model to the data and compares the fitted parameters to the known parameters.
    It repeats this process for different levels of correlation between x1 and x2, and returns the RMSE for each level
    of correlation.

    Args:
        x1: a 1D numpy array with the independent variable.
            Assume it is a 1D numpy array of floats with no NaNs or Infs.
        params:
            a 1D numpy array with the parameters of the model.
            Assume it is a 1D numpy array of three floats with no NaNs or Infs.
        pca_transform: If True, the data will be transformed using PCA before fitting the model.

    Returns:
        a 1D numpy array with the RMSE for each level of correlation.


    20 points

    """

    rmse_values = []
    for corr in np.linspace(0, 1, 50):
        x2 = generate_correlated_data(x1, corr)
        X = np.column_stack((np.ones(x1.shape[0]), x1, x2))
        if pca_transform:
            X = PCA_TRANS(X)
        y = X.dot(params) + np.random.normal(0, 1, x1.shape[0])
        model = LeastSquaresLinearRegression()
        model.fit(X, y)
        fitted_params = model.get_fitted_parameters()
        rmse = compute_rmse(params, fitted_params)
        rmse_values.append(rmse)
    return np.array(rmse_values)


def explain_effect_of_correlation_on_least_squares_no_pca():
    """
    Use the function you implemented in the previous question to explore the effect of correlation between the features
    as follows:
    - Generate a vector of 1000 random numbers from a normal distribution with mean 0 and standard deviation 1.
    - Run the function `explore_effect_of_correlation_on_least_squares` with the generated vector and two sets
     of parameters [1, 2, 3] and [-1, -2, 0]. Set `pca_transform=False`.

     Look at the returned RMSE values and identify the relationship between the correlation and the RMSE.
     Return a string with your observations (1-3 sentences). You should only return a string with your observations.
     This function should only return the answer: do all your processing elsewhere.

     Answer in English or Hebrew.

     How I grade:
        * I know the expected answer.
        * I provide the question, the expected answer, a rubric (how I grade),
            and your response to a Large Language Model (LLM) to get a grade and a feedback.
        * I ask the LLM to grade according to correctness and completeness.
        * Since the output of an LLM is not entirely predictable, I repeat the process three times and take
            the result with the highest grade.
        * I then manually check the result and adjust the grade if necessary.
        * You may answer in Hebrew (the model can handle that), but it will be easier if you answer in English.
            In any case, run your answer through a spelling and grammar checker.
        * Disputing the grade: If you think your answer was correct, you can dispute the grade.
            If you think that the reduction for partial mistakes or incompleteness was too high, it makes little sense
            to dispute the grade, save your disputes for when you think the grade is entirely wrong.

    20 points
    """
    return """
    With the PCA, the possibly correlated features are transformed
    into a new set of linearly uncorrelated components. With this, the root mean squared
    errors are much higher across all levels of correlation compared to wihtout PCA
    and they show an inrcrease for higher levels of correlation. 
    """


def explain_effect_of_correlation_on_least_squares_with_pca():
    """
    Use the function you implemented in the previous question to explore the effect of correlation between the features
    as follows:
    - Generate a vector of 1000 random numbers from a normal distribution with mean 0 and standard deviation 1.
    - Run the function `explore_effect_of_correlation_on_least_squares` with the generated vector and two sets
     of parameters [1, 2, 3] and [-1, -2, 0]. Set `pca_transform=True`.

     Look at the returned RMSE values and identify the relationship between the correlation and the RMSE.
     Compare the results to the results of the previous question (without PCA).

     Return a string with a discussion (1-3 sentences) about the similarity or the difference
     between the two sets of results. You should only return a string with your observations.

     This function should only return the answer: do all your processing elsewhere.
     Print nothing, only return the answer.

     Answer in English or Hebrew.

       How I grade:
        * I know the expected answer.
        * I provide the question, the expected answer, a rubric (how I grade),
            and your response to a Large Language Model (LLM) to get a grade and a feedback.
        * I ask the LLM to grade according to correctness and completeness.
        * Since the output of an LLM is not entirely predictable, I repeat the process three times and take
            the result with the highest grade.
        * I then manually check the result and adjust the grade if necessary.
        * You may answer in Hebrew (the model can handle that), but it will be easier if you answer in English.
            In any case, run your answer through a spelling and grammar checker.
        * Disputing the grade: If you think your answer was correct, you can dispute the grade.
            If you think that the reduction for partial mistakes or incompleteness was too high, it makes little sense
            to dispute the grade, save your disputes for when you think the grade is entirely wrong.


     20 points
    """
    return """
    Without the PCA, the model is exposed to raw data, which when the correlation between features increases
    the root mean squared error also tends to increase, especially more noticeable in the higher correlation values.
    """


def PCA_TRANS(data: np.ndarray) -> np.ndarray:
    data -= np.mean(data, axis=0)
    # Calculate the covariance matrix
    cov = np.cov(data, rowvar=False)
    # Calculate the eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Transform the data
    return data @ eigvecs

