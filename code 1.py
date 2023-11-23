import pandas as pd
import random
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import time

import data_manipulation


#Initializing...
training_size = 100
testing_size = 100




sampled_non_pulsars,sampled_pulsars,sampled_test_data,sampled_train_data,fraction_of_non_pulsars,fraction_of_pulsars = data_manipulation.normalize_dataset(training_size,testing_size)


fraction_of_pulsars = 0.5
fraction_of_non_pulsars = 0.5

# Define Serial Quantum Model

import quantum_circuit

import training


#Number of times the encoding gets repeated (here equal to the number of layers)
r = 8              
#Random initial weights
initial_weights = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True) 



#Finding the probability of finding pulsars and non_pulsars
probability_pulsar,probability_non_pulsar = training.pulsar_probability_training(sampled_non_pulsars,sampled_pulsars,initial_weights)
training.plot_probabilities(probability_pulsar,probability_non_pulsar,fraction_of_non_pulsars,fraction_of_pulsars,training_size,testing_size)




square_loss_choice = 0
cross_entropy_choice = 1
'''
# Call the training function with your data
max_epochs = 80
start_time = time.perf_counter()
optimized_weights_sqLoss,loss_array_sqLoss = training.training(max_epochs, initial_weights, sampled_train_data,square_loss_choice)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

opt_prob_pulsar,opt_prob_non_pulsar = training.pulsar_probability_training(sampled_non_pulsars,sampled_pulsars,optimized_weights_sqLoss)
training.plot_optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,fraction_of_non_pulsars,fraction_of_pulsars,training_size,testing_size,max_epochs,square_loss_choice)
training.plot_loss_function(fraction_of_non_pulsars,fraction_of_pulsars,training_size,testing_size,max_epochs,loss_array_sqLoss,square_loss_choice)
'''

#Optimizing Weights with Cross Entropy Loss

# Call the training function with your data
max_epochs = 80
start_time = time.perf_counter()
optimized_weights_crossEntropy,loss_array_crossEntropy = training.training(max_epochs, initial_weights, sampled_train_data,cross_entropy_choice)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

opt_prob_pulsar,opt_prob_non_pulsar = training.pulsar_probability_training(sampled_non_pulsars,sampled_pulsars,optimized_weights_crossEntropy)
training.plot_optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,fraction_of_non_pulsars,fraction_of_pulsars,training_size,testing_size,max_epochs,cross_entropy_choice)
training.plot_loss_function(fraction_of_non_pulsars,fraction_of_pulsars,training_size,testing_size,max_epochs,loss_array_crossEntropy,cross_entropy_choice)



'''
#Optimizing Weights with Cross Entropy Loss

# Call the training function with your data
epochs = 80
start_time = time.perf_counter()
optimized_weights_crossEntropy,loss_array_crossEntropy,epoch_stop = training(epochs, initial_weights, sampled_train_data,1)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

opt_prob_pulsar,opt_prob_non_pulsar = probability_of_pulsars_training(sampled_non_pulsars,sampled_pulsars,optimized_weights_crossEntropy)

#Plotting time
linspace_pulsar = np.linspace(0, int(fraction_of_non_pulsars*training_size), int(fraction_of_non_pulsars*training_size), requires_grad=False,dtype = int)/int(fraction_of_non_pulsars*training_size) #Linspace of samplesize 
plt.scatter(linspace_pulsar, opt_prob_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars

linspace_pulsar = np.linspace(0, int(fraction_of_pulsars*training_size), int(fraction_of_pulsars*training_size), requires_grad=False,dtype = int)/int(fraction_of_pulsars*training_size) #Linspace of samplesize 
plt.scatter(linspace_pulsar, opt_prob_pulsar, c='red', label = "Pulsars")  #Plots pulsars
plt.title("After Optimization - Cross Entropy Loss - Epoch {}".format(epochs))
plt.xlabel("Samples")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.show()

linspace_epoch = np.linspace(0, epochs, epochs, requires_grad=False,dtype = int) #Linspace of samplesize 
plt.plot(linspace_epoch, loss_array_crossEntropy, c='green', label = "Sq Loss")  #Plots pulsars
plt.title("Cross Entropy Loss against {} Epochs".format(epochs))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()

# Test data time:
#This is where we will run a new randomly sampled dataset in the quantum model with the optimized weights from the loss functions.

#Let's begin with the optimized weights from the square loss function.

pulsar_test_data = sampled_test_data[np.where(sampled_test_data[:, -1] == 1)[0]]
non_pulsar_test_data = sampled_test_data[np.where(sampled_test_data[:, -1] == 0)[0]]


#Applying the quantum model to datasets
expectation_val_Z_pulsar = [serial_quantum_model(optimized_weights_sqLoss, x_) for x_ in pulsar_test_data]
expectation_val_Z_non_pulsar = [serial_quantum_model(optimized_weights_sqLoss, x_) for x_ in non_pulsar_test_data]
#Unpacking.... (get rid of tensor...)
expectation_val_Z_pulsar = np.array([i.numpy() for i in expectation_val_Z_pulsar])
expectation_val_Z_non_pulsar = np.array([i.numpy() for i in expectation_val_Z_non_pulsar])
#Probability of getting a pulsar
probability_pulsar = (1+expectation_val_Z_pulsar)/2
#probability_non_pulsar = (1-expectation_val_Z_non_pulsar)/2

#Plotting time
linspace_pulsar_pulsar = np.linspace(0, len(pulsar_test_data), len(pulsar_test_data), requires_grad=False,dtype = int)/len(pulsar_test_data) #Linspace of samplesize 
linspace_pulsar_non_pulsar = np.linspace(0, len(non_pulsar_test_data), len(non_pulsar_test_data), requires_grad=False,dtype = int)/len(non_pulsar_test_data)
plt.scatter(linspace_pulsar_pulsar, probability_pulsar, c='red', label = "Pulsars")  #Plots pulsars
plt.scatter(linspace_pulsar_non_pulsar, probability_non_pulsar, c='blue', label = "Non-Pulsars")  #Plots pulsars
plt.axhline(y=.5, color='g', linestyle='--', label='Horizontal Line at probability = 0.5')
plt.title("Testing using Optimized weights from Square Loss")
plt.xlabel("Samples")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1.65, 0.5))
plt.ylim(0, 1)
plt.show()

#Here we set the horizontal line as our threshold

#Now, let's explore the optimized weights from the cross-entrropy loss function.

#Applying the quantum model to datasets
expectation_val_Z = [serial_quantum_model(optimized_weights_crossEntropy, x_) for x_ in sampled_test_data]
#Unpacking.... (get rid of tensor...)
expectation_val_Z = np.array([i.numpy() for i in expectation_val_Z])
#Probability of getting a pulsar
probability_pulsar = (1+expectation_val_Z)/2

#Plotting time
linspace_pulsar = np.linspace(0, testing_size, testing_size, requires_grad=False,dtype = int) #Linspace of samplesize 
plt.scatter(linspace_pulsar, probability_pulsar, c='red', label = "Pulsars")  #Plots pulsars

plt.title("Testing using Optimized weights from Cross Entropy Loss")
plt.xlabel("Samples")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.show()

# Error Section.

#Let's take the square loss to optimize the same set of data 5 different times and we will compare the weights.

normalized_dataset = normalize_columns(dataset)
sampled_non_pulsar_indices,sampled_pulsar_indices,sampled_test_indices,fraction_of_non_pulsars,fraction_of_pulsars = random_sample_with_condition(normalized_dataset, training_size,testing_size)
sampled_non_pulsars_errors = normalized_dataset[sampled_non_pulsar_indices]
sampled_pulsars_errors = normalized_dataset[sampled_pulsar_indices]
sampled_train_data_errors = np.vstack([sampled_non_pulsars,sampled_pulsars]) #Joining the two random samples

#Random initial weights
initial_weights_error1 = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True) 
initial_weights_error2 = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True)
initial_weights_error3 = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True)
initial_weights_error4 = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True)
initial_weights_error5 = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True)
# Call the training function with your data
epochs = 50
start_time = time.perf_counter()
optimized_weights_sqLoss_error1,loss_array_sqLoss_error1 = training(epochs, initial_weights_error1, sampled_train_data_errors,0)
optimized_weights_sqLoss_error2,loss_array_sqLoss_error2 = training(epochs, initial_weights_error2, sampled_train_data_errors,0)
optimized_weights_sqLoss_error3,loss_array_sqLoss_error3 = training(epochs, initial_weights_error3, sampled_train_data_errors,0)
optimized_weights_sqLoss_error4,loss_array_sqLoss_error4 = training(epochs, initial_weights_error4, sampled_train_data_errors,0)
optimized_weights_sqLoss_error5,loss_array_sqLoss_error5 = training(epochs, initial_weights_error5, sampled_train_data_errors,0)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

print("optimized_weights_sqLoss_error1 = ",optimized_weights_sqLoss_error1)
print("optimized_weights_sqLoss_error2 = ",optimized_weights_sqLoss_error2)
print("optimized_weights_sqLoss_error3 = ",optimized_weights_sqLoss_error3)
print("optimized_weights_sqLoss_error4 = ",optimized_weights_sqLoss_error4)
print("optimized_weights_sqLoss_error5 = ",optimized_weights_sqLoss_error5)

print(initial_weights_error1)

#Now we will use 5 randomly sampled datasets for the same initial weights.

num_iterations = 5
epochs = 50
optimized_weights_sqLoss_array = []
initial_weights_error = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True) 
# Loop through the desired number of iterations
start_time = time.perf_counter()
for n in range(1, num_iterations + 1):
    # Your existing code
    normalized_dataset = normalize_columns(dataset)
    sampled_non_pulsar_indices, sampled_pulsar_indices, sampled_test_indices, fraction_of_non_pulsars, fraction_of_pulsars = random_sample_with_condition(normalized_dataset, training_size, testing_size)
    sampled_non_pulsars_errors = normalized_dataset[sampled_non_pulsar_indices]
    sampled_pulsars_errors = normalized_dataset[sampled_pulsar_indices]
    sampled_train_data_errors_n = np.vstack([sampled_non_pulsars, sampled_pulsars])
    # Call the training function with your data
    optimized_weights_sqLoss_errorn,loss_array_sqLoss_errorn = training(epochs, initial_weights_error, sampled_train_data_errors_n,0)
    optimized_weights_sqLoss_array.append(optimized_weights_sqLoss_errorn)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

# Calculate the mean weights for each feature
mean_weights = np.mean(optimized_weights_sqLoss_array, axis=0)

# Calculate the standard deviation of the weights for each feature
std_dev_weights = np.std(optimized_weights_sqLoss_array, axis=0)

# Now, mean_weights contains the mean weight for each of the 8 features,
# and std_dev_weights contains the standard deviation (error) for each feature's weight.

# You can access the mean and standard deviation for each feature as follows:
for feature_index, (mean, std_dev) in enumerate(zip(mean_weights, std_dev_weights)):
    print(f"Feature {feature_index}: Mean Weight = {mean}, Standard Deviation = {std_dev}")



'''