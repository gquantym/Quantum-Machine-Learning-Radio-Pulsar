# Test data time:
#This is where we will run a new randomly sampled dataset in the quantum model with the optimized weights from the loss functions.

#Let's begin with the optimized weights from the square loss function.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def testing_data(sampled_test_data):
    pulsar_test_data = sampled_test_data[np.where(sampled_test_data[:, -1] == 1)[0]]
    non_pulsar_test_data = sampled_test_data[np.where(sampled_test_data[:, -1] == 0)[0]]
    return pulsar_test_data,non_pulsar_test_data

def plot_test_probabilities(probabilities,classifications,epoch,loss_func_choice):
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

    sensitivity,specificity,threshold = calculate_sensitivity_specificity(probabilities,classifications)
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

def find_optimal_threshold(y_prob, y_true):
    """
    Find the optimal threshold that maximizes the F1 score.

    Parameters:
    - y_true: True labels (1 for pulsar, 0 for non-pulsar)
    - y_prob: Predicted probabilities

    Returns:
    - optimal_threshold: Optimal threshold for classification
    """
    thresholds = np.linspace(0, 1, num=11)
    f1_scores = [f1_score(y_true, (y_prob > t).astype(int)) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print("Optimal Threshold is == ",optimal_threshold)

    return optimal_threshold
    
def calculate_sensitivity_specificity(probabilities,classifications):
    """
    Calculate Sensitivity and Specificity.

    Parameters:
    - true_positive: Number of true positive cases
    - false_negative: Number of false negative cases
    - false_positive: Number of false positive cases
    - true_negative: Number of true negative cases

    Returns:
    - sensitivity: Sensitivity (True Positive Rate)
    - specificity: Specificity
    """
    
    threshold = find_optimal_threshold(probabilities,classifications)
    
    # Create a boolean mask for class 0 and class 1
    class_0_mask = (classifications == 0)
    class_1_mask = (classifications == 1)
    # Use boolean indexing to get features for class 0 and class 1
    predictions_non_pulsar = probabilities[class_0_mask]
    predictions_pulsar = probabilities[class_1_mask]
    
    # Apply threshold to get binary predictions
    binary_predictions_pulsar = (predictions_pulsar < threshold).astype(int)
    binary_predictions_non_pulsar = (predictions_non_pulsar > threshold).astype(int)
    
    # Calculate confusion matrix
    true_positive = np.sum((binary_predictions_pulsar == 0))
    false_negative = np.sum((binary_predictions_pulsar == 1))
    false_positive = np.sum((binary_predictions_non_pulsar == 1))
    true_negative = np.sum((binary_predictions_non_pulsar == 0))

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    # Convert to float if the result is a numpy array
    sensitivity = float(sensitivity) if isinstance(sensitivity, np.ndarray) else sensitivity
    specificity = float(specificity) if isinstance(specificity, np.ndarray) else specificity
    return sensitivity, specificity, threshold
