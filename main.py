from pennylane import numpy as np
import time
import training
import data
import plot

def main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_set,quantum_circuit):
    global weights_crossEntropy,probability_non_pulsar,probability_pulsar,train_data
    '''Initializing constants'''
    square_loss_choice = 0
    cross_entropy_choice = 1
    '''Initializing constants'''
    #Finding the probability of finding pulsars and non_pulsars
    probability_non_pulsar = training.pulsar_probability(non_pulsar_samples[:,:8],initial_weights,quantum_circuit)
    probability_pulsar = training.pulsar_probability(pulsar_samples[:,:8],initial_weights,quantum_circuit)
    plot.probabilities(probability_pulsar,probability_non_pulsar)
      
    #----------------------------NOW LETS TRAIN DATA----------------------------
    
    train_data = np.vstack((pulsar_samples, non_pulsar_samples))
    #Optimizing Weights with Cross Entropy Loss
    start_time = time.perf_counter()
    weights_crossEntropy,loss_crossEntropy = training.training(epochs, initial_weights, train_data, cross_entropy_choice,quantum_circuit)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    
    opt_prob_non_pulsar = training.pulsar_probability(non_pulsar_samples[:,:8],weights_crossEntropy,quantum_circuit)
    opt_prob_pulsar = training.pulsar_probability(pulsar_samples[:,:8],weights_crossEntropy,quantum_circuit)
    plot.optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epochs,cross_entropy_choice)
    plot.loss_function(epochs,loss_crossEntropy,cross_entropy_choice)
    
    #----------------------------NOW LETS TEST DATA----------------------------

    test_features,test_class = data.feature_class_split(test_set)
    test_probabilities = training.pulsar_probability(test_features,weights_crossEntropy,quantum_circuit)
    plot.test_probabilities(test_probabilities,test_class,epochs,cross_entropy_choice)


#import quantum_repeated as quantum_circuit
#Number of times the encoding gets repeated (here equal to the number of layers)
#import quantum_circuit as quantum_circuit
r = 8   
#initial_weights = 2 * np.pi * np.random.random(size=(3,r+1, 2))#, requires_grad=True) 
initial_weights = 2 * np.pi * np.random.random(size=(r+1, 3))#, requires_grad=True) 
epochs = 120

normalized_dataset = data.normalize()
    
pulsar_samples,non_pulsar_samples,test_set = data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)



import matplotlib.pyplot as plt
import numpy as np

# Assuming 'sampled_data' is your input data
sampled_data = np.random.rand(100, 8)  # Replace this with your actual data

# Plot histograms for each feature
num_features = sampled_data.shape[1]

plt.figure(figsize=(15, 10))

for i in range(num_features-1):
    plt.subplot(2, 4, i + 1)  # Adjust the subplot grid based on your number of features
    plt.hist(sampled_data[:, i], bins=30, color='blue', alpha=0.7)
    plt.title(f'Feature {i + 1}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


import old_quantum_circuit as quantum_circuit
main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_set,quantum_circuit)



'''
#import quantum_circuit_improved as quantum_circuit
#main(initial_weights[0],epochs,pulsar_samples,non_pulsar_samples,test_set,quantum_circuit)
import mo_quantum_circuit as quantum_circuit
#print("-----------------------------------------------FIRST COMPLETE---------------------------")
main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_set,quantum_circuit)
'''
