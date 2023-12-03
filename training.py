# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:06:54 2023

@author: gdlpi
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
'''
def pulsar_probability(sampled_data,weights,quantum_circuit): 
    expectation_value = [quantum_circuit.serial_model(weights, x_) for x_ in sampled_data]
    #Code below unpacks the contents of expectation_val_Z_zeros & expectation_val_Z_ones from tensor form to np array!
    expectation_value = np.array([i.numpy() for i in expectation_value])
    probability_pulsar = (1-expectation_value)/2
    return probability_pulsar'''

def pulsar_probability(sampled_data, weights, quantum_circuit):
    global expectation_value
    expectation_value = [quantum_circuit.serial_model(weights, x_) for x_ in sampled_data]
    probability_pulsar = (1 - np.array(expectation_value)) / 2
    return probability_pulsar


def plot_probabilities(probability_pulsar,probability_non_pulsar):
    linspace_non_pulsar = np.linspace(0, len(probability_non_pulsar), len(probability_non_pulsar),dtype = int)/len(probability_non_pulsar) #Linspace of samplesize 
    linspace_pulsar = np.linspace(0, len(probability_pulsar), len(probability_pulsar),dtype = int)/len(probability_pulsar) #Linspace of samplesize 
    print("linspace_non_pulsar = ",len(linspace_non_pulsar))
    print("linspace_pulsar = ",len(linspace_pulsar))
    print("Probability pulsar = ",len(probability_pulsar))
    print("Probability non_pulsar = ",len(probability_non_pulsar))
    plt.scatter(linspace_non_pulsar, probability_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, probability_pulsar, c='red', label = "Pulsars")  #Plots pulsars
    plt.title("Before Optimization")
    plt.xlabel("Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1, 0.5))
    #plt.ylim(0, 1)
    plt.show()
    
#Loss/Cost functions

def cross_entropy_loss(weights, data,quantum_circuit):
    predictions = [quantum_circuit.serial_model(weights, x) for x in data]
    predictions = np.array(predictions)
    predictions = (1-predictions)/2
    targets = np.array([row[-1] for row in data])
    targets = targets.reshape(predictions.shape)
    # Clip predicted probabilities to avoid log(0) issues
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    # Compute the cross-entropy loss
    loss = -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    loss = loss / len(targets)
    return loss

def square_loss(weights, data,quantum_circuit):
    loss = 0
    predictions = [quantum_circuit.serial_model(weights, x) for x in data]
    predictions = np.array(predictions)
    predictions = (1-predictions)/2
    targets = np.array([row[-1] for row in data])
    for t, p in zip(targets, predictions):
        loss += (t - p) ** 2
    loss = loss / len(targets)
    return 0.5*loss #Normalizing distance

# Machine Learning Section. 

#Optimization using built-in PennyLane optimizer (Adam). stepsize=0.1: This parameter sets the learning rate for the Adam optimizer. The learning rate controls the step size during optimization and affects how quickly the model's parameters are updated

def training(epochs, initial_weights, sampled_train_data, loss_func_choice,quantum_circuit): #loss_func_choice 0 if we want square loss 1 if we want 
    global optimized_weights
    optimizer = qml.AdamOptimizer(stepsize=0.1)
    loss_array = np.array([])
    # Define the loss function based on choice
    if loss_func_choice == 0:
        loss_function = square_loss
    else:
        loss_function = cross_entropy_loss
    for epoch in range(epochs):
   
        optimized_weights, loss= optimizer.step_and_cost(lambda w: loss_function(w, sampled_train_data,quantum_circuit), initial_weights)
        initial_weights = optimized_weights
        loss_array = np.append(loss_array, loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.8f}")
    return optimized_weights,loss_array

def plot_optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epoch,loss_func_choice):
    #Plotting time
    linspace_non_pulsar = np.linspace(0, len(opt_prob_non_pulsar), len(opt_prob_non_pulsar),dtype = int)/len(opt_prob_non_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_non_pulsar, opt_prob_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    linspace_pulsar = np.linspace(0, len(opt_prob_pulsar), len(opt_prob_pulsar),dtype = int)/len(opt_prob_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_pulsar, opt_prob_pulsar, c='red', label = "Pulsars")  #Plots pulsars
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.title("After Optimization - {0} - Epoch {1}".format(titles[loss_func_choice],epoch))
    plt.xlabel("Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.ylim(0, 1)
    plt.show()

def plot_loss_function(epoch,loss_array,loss_func_choice):
    linspace_epoch = np.linspace(0, epoch, epoch,dtype = int) #Linspace of samplesize 
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.plot(linspace_epoch, loss_array, c='green', label = "{}".format(titles[loss_func_choice]))  #Plots pulsars
    plt.title("{0} against - Epoch {1}".format(titles[loss_func_choice],epoch))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.show()

