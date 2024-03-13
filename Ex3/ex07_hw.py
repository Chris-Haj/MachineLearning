# SVD and PCA homework
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA

"""
Submission instructions:
    - only submit the `ex04_hw.py` file. Don't zip it.
    - Make sure to include your id number in the `get_id_number` function.
    - Make sure to remove any code that is not part of the functions you are implementing.
    - Do not change the function signatures.
    - Do not add any additional imports
    
"""


def get_id_number() -> str:
    return '207824772'


def get_number_of_features_that_explain_variance(
        df: pd.DataFrame, explained_variance_ratio: float
) -> (int, int):
    """
    Use SVD and PCA to identify the number of features that explain at least
    `explained_variance` of the variance in the dataset.

    You may assume that all the columns in the dataset are numeric.

    Args:
        df: DataFrame with the dataset to analyze.
        explained_variance_ratio: float between 0 and 1.
        The minimum amount of variance ratio that the number of features should explain.

    Returns:
        a tuple of two integers: (n_features_svd, n_features_pca)

    30 points
    """

    raise NotImplementedError()


def compute_rmse(d1, d2) -> float:
    """
    Compute the root mean square error between two dataframes.
    In this context, RMSE is defined as
    RMSE = sqrt(1/n * sum((df1 - df2)^2))

    Args:
        d1: DataFrame or a numpy array
        d2: DataFrame or a numpy array

    Returns:
        float: the RMSE between d1 and d2

    5 points
    """
    raise NotImplementedError()


def compute_reconstructed_error_svd(data: np.ndarray, n_components: int) -> float:
    """
    Compute the error of the reconstructed data using SVD.

    Args:
        data: a numpy array or a DataFrame with the original data
        n_components: how many components to use for the reconstruction

    Returns:
        float: the RMSE between the original data and the reconstructed data

    10 points
    """
    raise NotImplementedError()


def compute_reconstructed_error_pca(data: np.ndarray, n_components: int) -> float:
    """
    Compute the error of the reconstructed data using PCA.

    Args:
        data: a numpy array or a DataFrame with the original data
        n_components: how many components to use for the reconstruction

    Returns:
        float: the RMSE between the original data and the reconstructed data

    10 points
    """
    raise NotImplementedError()


def compute_k_svd_raw() -> int:
    """
    You are given a dataset in `data_hw.csv`. All the columns, except for `original_title` are numeric.
    Use pandas to read the dataset, discard the `original_title` column.

    * Use SVD to identify the number of features that explain at least 90% of the relative variance in the dataset.

    Returns:
        int: the number of features that explain at least 90% of the variance in the dataset.

    5 points
    """
    raise NotImplementedError()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each column by subtracting the mean and dividing by the standard deviation.

    Args:
        df: DataFrame with the dataset to normalize.

    Returns:
        DataFrame: the normalized dataset

    5 points
    """
    raise NotImplementedError()


def compute_k_svd_normalized() -> int:
    """
    You are given a dataset in `data_hw.csv`. All the columns, except for `original_title` are numeric.

    Use pandas to read the dataset, discard the `original_title` column.

    * Normalize each colum using the `normalize_columns` function.

    * Use SVD to identify the number of features that explain at least 90% of the relative variance in the dataset.
    Returns:
        int: the number of features that explain at least 90% of the variance in the dataset.

    5 points
    """
    raise NotImplementedError()


def explain_the_different_ks():
    """
    Explain the difference between the K you found in `compute_k_svd_raw` and `compute_k_svd_normalized`.
    If there is a difference, explain why. If there is no difference, explain why not.

    Return your answer as a string.
    30 points
    """
    explanation = """
    
    """
    raise NotImplementedError()
    return explanation


if __name__ == "__main__":
    assert isinstance(get_id_number(), str)
    id_number = int(get_id_number())
    assert str(id_number) == get_id_number()
    df = pd.read_csv('data_hw.csv').head(15)
    string_col = df.iloc[:, 0]
    df_numeric = df.drop(df.columns[0], axis=1)
    data = df_numeric.to_numpy()
    dataMean = np.mean(data, axis=0)
    dataSTD = np.std(data, axis=0)
    standardizedData = (data - dataMean) / dataSTD
    standardizedDF = pd.DataFrame(standardizedData, columns=df_numeric.columns)
    standardizedDF.insert(0, df.columns[0], string_col)
    print(standardizedDF[0:5,0:5])