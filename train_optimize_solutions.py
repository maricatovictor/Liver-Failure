from scipy import stats
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import math

def mean_pred(y_true, y_pred): #Loss function for problem error evaluation on TensorFlow
    RMSE = tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred))
    return RMSE

#Get Train Data (Conditions)
x_sheet = pd.read_excel('coded_data.xlsx', usecols=[1, 2, 3, 4, 9, 10, 11])
df = np.asmatrix(x_sheet.values)  # Make it a npmatrix

#Get Train Data (Results)
y_sheet = pd.read_excel('coded_data.xlsx', usecols=[7])
labels = np.array(y_sheet.values)

from sklearn.model_selection import train_test_split
#Split Data - train_test_split simple splitting
X_train, X_test, y_train, y_test = train_test_split(
    df, labels, test_size=0.2, random_state=89) #Split data to 20% test and 80% training

#Fit a linear regressor (sklearn) for predicted x real evaluation (R2, etc.)
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(y_test, y_test)

shape = np.shape(X_train) #Shape of train array (number of events, no of variables)
num_var = shape[1] #how many variables on training data

# Neural Network Structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(num_var + 1, activation='linear'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.7, seed=36),
    tf.keras.layers.Dense(20, activation='selu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Neural Network Parameters
lr = 0.002
optimizer = tf.keras.optimizers.Nadam(lr=lr)
model.compile(
    optimizer=optimizer,
    loss=mean_pred)

    # Neural Network Fitting and Validation data
model.fit(X_train, y_train, epochs=200, steps_per_epoch=(int(len(X_train))),
          validation_data=(X_test, y_test), validation_steps=(int(len(X_test))))

test_ans = model.predict(X_test) #predict the test data and create an array of results

tensor_RMSE = math.sqrt(mean_squared_error(y_test, test_ans)) #get RMSE from real x predicted
print("RMSE: " + str(tensor_RMSE))

r2_score = linear_reg.score(y_test, test_ans) #print r2 score from the predicted results vs the real data
print("R2 Score from RxR + Scatter Predicted: %f" %r2_score)

slope, intercept, r_value, p_value, std_err = stats.linregress(
        y_test[:, 0], test_ans[:, 0]) #get some constants and results from the regression line generated from the neural network
print("R2 Score from predicted x real %f" %r_value**2) #print the r2 of predicted value x fitted line on ann


plt.scatter(y_test, test_ans) #plot the predicted values (y) x real(x)
plt.plot(y_test, y_test, 'black', label='real values') #plot the alpha=1 line (real x real)
plt.plot(y_test, intercept + slope * y_test, 'r', label='fitted line') #plot the ann regression line
plt.xlabel('Real values')
plt.ylabel('Predicted values')


import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Regression from model')
black_patch = mpatches.Patch(color='black', label='Regression from real data')
plt.legend(handles=[red_patch, black_patch])
#intercept + slope * y_test

#plt.savefig('model_test_1.png') #save the generated figure
plt.show() #show graphic

import random
data = X_train
fitnesses = [1]
def fitness (individual, data):
    fitness = 0
    #print(individual)
    #print(np.shape(individual))
    ind_predicted = model.predict(individual)[0] #get the predicted value from ann
    fitness = abs((ind_predicted - intercept)/slope) #convert the value from regression fit
    #print(fitness)
    if(fitnesses[-1]>fitness): #if the fitness from the current individual analyzed is smaller than last one, add it to an array
        fitnesses.append(fitness) #This array is used to plot the generation graphics
    return fitness

#create initial population with limited range (avoid outliers and exaggerated parameters), keep it real
def create_individual(data):
    dmso = random.uniform(0, 0.25)
    sfb = random.uniform(0, 1)
    meio_cultura = random.uniform(0,1)
    tipo_meio = random.randint(1,5)
    trehalose = random.uniform(0,0.5)
    glic = random.uniform(0,0.5)
    sac = random.uniform(0,0.5)
    return np.array([[dmso, sfb, meio_cultura, tipo_meio, trehalose, glic, sac]])
print(create_individual(data))

def crossover(parent_1, parent_2):
    #crossover_index = np.random.uniform(0,1, len(parent_1))
    crossover_index = np.random.randint(0, np.shape((parent_1))[1]) #How many variables will come from parent 1 or 2
    #print(parent_1);print(parent_2)
    #print(crossover_index)
    #print("Parent1:");print(parent_1[0][:crossover_index])
    #print("Parent2:");print(parent_2[0][crossover_index:])
    #The crossing over will generate 2 individuals: one with the first values form parent 1 and rest with values from p2 & another with first values from parent2
    child_1 = np.array([parent_1[0][:crossover_index]])
    child_1 = np.append(child_1, parent_2[0][crossover_index:])
    child_2 = np.array([parent_2[0][:crossover_index]])
    child_2 = np.append(child_2, parent_1[0][crossover_index:])
    #Reshape the arrays for the ann used shape
    child_1 = child_1.reshape(1,num_var)
    child_2 = child_2.reshape(1,num_var)
    #print("Child1:");print(child_1)
    #print("Child2:");print(child_2)
    return child_1, child_2

# For the mutate function, supply one individual (i.e. a candidate solution representation) as a parameter:
def mutate(individual):
    #print(individual[0])
    mutate_index = random.randrange(np.shape(individual)[0])
    if individual[0][mutate_index] == 0:
        individual[0][mutate_index] == 1
    else:
        individual[0][mutate_index] == 0

pop = 1000
from pyeasyga.pyeasyga import GeneticAlgorithm
ga = GeneticAlgorithm(data, population_size=pop,
                               generations=150,
                               crossover_probability=0.2,
                               mutation_probability=0.1,
                               elitism=False,
                               maximise_fitness=False)

ga.create_individual = create_individual
ga.fitness_function = fitness
ga.crossover_function = crossover
ga.mutate_function = mutate

ga.run()
best_ind = ga.best_individual()
print(best_ind)

#Create a graphics for fitness behaviour during GA execution
x_axis = []
i = 0
for index in range(len(fitnesses)):
    i += 1 #for each parameters in fitness value, get an index value on an array that will be used to plot X axis on graphic
    x_axis.append(i)
print(x_axis)
print(fitnesses)

#Graphics from fitness decrease during generation
plt.scatter(x_axis, fitnesses)
plt.ylabel('Fitness')
plt.show()
plt.savefig('ga_1.png')

#Print the parameters from the best individual on an organized and understandable way
dmso = best_ind[1][0][0]*100
sfb = best_ind[1][0][1]*100
conc_meio = best_ind[1][0][2]*100
meio = best_ind[1][0][3]
treh = best_ind[1][0][4]*100
glic = best_ind[1][0][5]*100
sac = best_ind[1][0][6]*100
print(" Optimized: \n DMSO:%f \n SFB: %f \n MEDIUM CONCENTRATION: %f\n MEDIUM TYPE: %f\n Trehalose: %f\n Glycerol %f\n Sucrose: %f\n" %(dmso, sfb, conc_meio, meio, treh, glic, sac))
