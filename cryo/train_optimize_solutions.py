
from scipy import stats
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from mlp_nn import *
from evaluate import *


def main():

    x_sheet = pd.read_excel('../data/coded_data.xlsx', usecols=[1, 2, 3, 4, 9, 10, 11])
    df = np.asmatrix(x_sheet.values)

    y_sheet = pd.read_excel('../data/coded_data.xlsx', usecols=[7])
    labels = np.array(y_sheet.values)

    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.2, random_state=89)

    n_variables = num_var(X_train)
    model, test_ans = train_nn(X_train, y_train, X_test, y_test, n_variables)

    intercept, slope = evaluate(y_test, test_ans)

    model.save("models/model.h5")

    # from genetic_algorithm import *
    # genetic_algorithm(X_train, model, num_var, intercept, slope)


if __name__ == '__main__':
    main()
