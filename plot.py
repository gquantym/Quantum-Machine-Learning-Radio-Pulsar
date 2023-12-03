import numpy as np
import matplotlib.pyplot as plt

def optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epoch,loss_func_choice):
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

def loss_function(epoch,loss_array,loss_func_choice):
    linspace_epoch = np.linspace(0, epoch, epoch,dtype = int) #Linspace of samplesize 
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.plot(linspace_epoch, loss_array, c='green', label = "{}".format(titles[loss_func_choice]))  #Plots pulsars
    plt.title("{0} against - Epoch {1}".format(titles[loss_func_choice],epoch))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.show()

def probabilities(probability_pulsar,probability_non_pulsar):
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
    

def test_probabilities(probabilities,classifications,epoch,loss_func_choice):
    import testing
    # Create a boolean mask for class 0 and class 1
    class_0_mask = (classifications == 0)
    class_1_mask = (classifications == 1)
    # Use boolean indexing to get features for class 0 and class 1
    non_pulsar_probability = probabilities[class_0_mask]
    pulsar_probability = probabilities[class_1_mask]
    linspace_pulsar = np.linspace(0, len(pulsar_probability), len(pulsar_probability),dtype = int)/len(pulsar_probability) #Linspace of samplesize 
    linspace_non_pulsar = np.linspace(0, len(non_pulsar_probability), len(non_pulsar_probability),dtype = int)/len(non_pulsar_probability)
    plt.scatter(linspace_non_pulsar, non_pulsar_probability, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, pulsar_probability, c='red', label = "Pulsars")  #Plots pulsars

    sensitivity,specificity,threshold = testing.calculate_sensitivity_specificity(probabilities,classifications)
    print("Sensitivity = {}%".format(sensitivity*100))
    print("Specificity = {}%".format(specificity*100))
    
    plt.axhline(y=threshold, color='g', linestyle='--', label='Horizontal Line at probability = {:.1f}\nSensitivity = {:.1f}%\nSpecificity = {:.1f}%'.format(threshold,sensitivity*100,specificity*100))
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.title("Test data using weights from {0} - Epoch {1}".format(titles[loss_func_choice],epoch))
    plt.xlabel("Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)
    plt.show()