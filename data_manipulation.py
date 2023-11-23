import pandas as pd
import random
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import time

# Define a function to normalize the data to the range [-π, π]
def normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data_normalized = [(-np.pi + (2 * np.pi * (x - min_val) / (max_val - min_val))) for x in data]
    return data_normalized

def normalize_columns(data):
    # Get the number of columns (features)
    num_columns = data.shape[1]
    # Create an array of each column
    column_arrays = [data[:, i] for i in range(num_columns)] #NUM COLUMNS-1 TO REMOVE CLASSIFICATION.
    classification_array = column_arrays[8]
    # Now, column_arrays is a list of NumPy arrays, where each array represents a column from the original dataset.
    normalized_column_data = np.array([])
    for i in column_arrays:
        normalized_column = normalization(column_arrays)
        normalized_column_data = np.vstack(normalized_column)
    #use np.vstack to stack the arrays vertically and then transpose to get the original dataset
    normalized_dataset = np.vstack(normalized_column_data).T
    if len(classification_array) != normalized_dataset.shape[0]:
        raise ValueError("Number of elements in the 1D array must match the number of rows in the 2D array.")
    # Append the 1D array to the end of each row in the 2D array
    return np.hstack((normalized_dataset, classification_array[:, np.newaxis])) #Now we've added the classification columns to the normalized dataset.



def random_sample_with_condition(normalized_dataset, training_size, testing_size):
    # Create an array of indices where the 8th value is either 0 or 1
    #condition_indices = np.where((normalized_dataset_2[:, 8] == 0) | (normalized_dataset_2[:, 8] == 1))[0]
    non_pulsar_indices = np.where(normalized_dataset[:, -1] == 0)[0] #Finds index of where the last value in array =1
    
    pulsar_indices = np.where(normalized_dataset[:, -1] == 1)[0]
    
    fraction_of_pulsars = 0.5 #len(pulsar_indices)/(len(non_pulsar_indices)+len(pulsar_indices))
    fraction_of_non_pulsars = 0.5 #len(non_pulsar_indices)/(len(non_pulsar_indices)+len(pulsar_indices))
    #print(fraction_of_non_pulsars)
    #print("int(training_size*fraction_of_pulsars) = ",int(training_size*fraction_of_pulsars))
    #print("int(training_size*fraction_of_non_pulsars) = ",int(training_size*fraction_of_non_pulsars))
    
    # Randomly sample indices that satisfy the condition
    sampled_non_pulsar_indices = np.random.choice(non_pulsar_indices, int(training_size*fraction_of_non_pulsars), replace=False)
    sampled_pulsar_indices = np.random.choice(pulsar_indices, int(training_size*fraction_of_pulsars), replace=False)
    
    #NEEED TO MAKE A WAY TO FIND DIFFERENCE IN INDICES FROM MAIN DATASET AND THE SAMPLED AVOVE
    # Create a list of available indices for testing data after excluding training indices
    sampled_train_indices = np.append(sampled_non_pulsar_indices,sampled_pulsar_indices)
    
    available_indices = np.setdiff1d(np.arange(normalized_dataset.shape[0]), sampled_train_indices)
    
    # Sample indices for testing data without replacement from available_indices
    sampled_test_indices = np.random.choice(available_indices, testing_size, replace=False)
    

    return sampled_non_pulsar_indices,sampled_pulsar_indices,sampled_test_indices, fraction_of_non_pulsars, fraction_of_pulsars




def normalize_dataset(training_size,testing_size):
    dataset = np.genfromtxt('pulsar.csv', delimiter = ',', skip_header=1)
    normalized_dataset = normalize_columns(dataset)


    sampled_non_pulsar_indices,sampled_pulsar_indices,sampled_test_indices,fraction_of_non_pulsars,fraction_of_pulsars = random_sample_with_condition(normalized_dataset, training_size,testing_size)

    sampled_non_pulsars = normalized_dataset[sampled_non_pulsar_indices]
    sampled_pulsars = normalized_dataset[sampled_pulsar_indices]
    sampled_test_data = normalized_dataset[sampled_test_indices]
    sampled_train_data = np.vstack([sampled_non_pulsars,sampled_pulsars]) #Joining the two random samples
    return sampled_non_pulsars,sampled_pulsars,sampled_test_data,sampled_train_data,fraction_of_non_pulsars,fraction_of_pulsars