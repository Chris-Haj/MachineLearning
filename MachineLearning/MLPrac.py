import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


def createDataSet():
    data = pd.read_csv('Data.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return data, x, y


if __name__ == '__main__':
    data, x, y = createDataSet()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(x[:, 1: 3])
    print(x)
    print()
    x[:, 1:3] = imputer.transform(x[:, 1:3])
    columnTrans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    print(x)
    print()
    x = np.array(columnTrans.fit_transform(x))
    print(x)
    le = LabelEncoder()
    y = le.fit_transform(y)
    xTrain , xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2 )
