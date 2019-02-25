from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
import math

def evaluate(y_test, test_ans):
    #Fit a linear regressor (sklearn) for predicted x real evaluation (R2, etc.)
    linear_reg = LinearRegression()
    linear_reg.fit(y_test, y_test)

    r2_score = linear_reg.score(y_test, test_ans) #print r2 score from the predicted results vs the real data
    print("R2 Score from RxR + Scatter Predicted: %f" %r2_score)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
            y_test[:, 0], test_ans) #get some constants and results from the regression line generated from the neural network
    print("R2 Score from predicted x real %f" %r_value**2) #print the r2 of predicted value x fitted line on ann


    plt.scatter(y_test, test_ans) #plot the predicted values (y) x real(x)
    plt.plot(y_test, y_test, 'black', label='real values') #plot the alpha=1 line (real x real)
    plt.plot(y_test, intercept + slope * y_test, 'r', label='fitted line') #plot the ann regression line
    plt.xlabel('Real values')
    plt.ylabel('Predicted values')



    red_patch = mpatches.Patch(color='red', label='Regression from model')
    black_patch = mpatches.Patch(color='black', label='Regression from real data')
    plt.legend(handles=[red_patch, black_patch])
    #intercept + slope * y_test

    #plt.savefig('model_test_1.png') #save the generated figure
    plt.show() #show graphic

    return intercept, slope