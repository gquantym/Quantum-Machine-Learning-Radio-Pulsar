import pennylane as qml
from pennylane import numpy as np
dev = qml.device('default.qubit', wires=1)
def S(x,theta):
    """Data-encoding circuit block."""
    for i in range(len(x)-1): #WE CHANGED THIS WATCH OUT
        qml.RZ(x[i] + theta[1], wires=0)

def W(theta):
    """Trainable circuit block."""
    qml.RY(theta[0], wires=0)

 
@qml.qnode(dev, interface="autograd")
def serial_model(weights, x):
    for theta in weights[:-1]:  #FOR THETA IN ALL THE WEIGHTS UP TO (BUT NOT INCLUDING) THE LAST ONE 
        #print(theta)    
        W(theta)
        S(x,theta)
    # (L+1)'th unitary.
    W(weights[-1]) #Now this is the last W gate as the input is the last element in the weight array.
    #THIS BELOW IS JUST TO TEST IF ITS CLASSIFYING...
    #qml.RY(np.pi/2,wires=0) 
    #qml.RZ(np.pi*(1-x[8]),wires=0)    
    #qml.RY(np.pi/2,wires=0)
    
    return qml.expval(qml.PauliZ(wires=0)) #Expectation value of the what??
