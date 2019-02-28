
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


#Get Train Data (Conditions)
x_sheet = pd.read_excel('coded_data.xlsx', usecols=[1, 2, 3, 4, 9, 10, 11])
df = np.asmatrix(x_sheet.values)  # Make it a npmatrix

#Get Train Data (Results)
y_sheet = pd.read_excel('coded_data.xlsx', usecols=[7])
labels = np.array(y_sheet.values)

##%% 
#Split Data - train_test_split simple splitting
X_train, X_test, y_train, y_test = train_test_split(
    df, labels, test_size=0.2, random_state=89) #Split data to 20% test and 80% training

from mlp_nn import *
#model, test_ans = train_nn(X_train, y_train, X_test, y_test)
num_var = num_var(X_train)

from svm import *
clf = svm(X_train, y_train)
test_ans = clf.predict(X_test)


from evaluate import *
intercept, slope = evaluate(y_test, test_ans)

from genetic_algorithm import *
genetic_algorithm(X_train, model, num_var, intercept, slope)