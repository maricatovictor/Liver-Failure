import random
import matplotlib.pyplot as plt
import numpy as np
def genetic_algorithm(data, model, num_var, intercept, slope):


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
