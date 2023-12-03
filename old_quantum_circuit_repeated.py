import pennylane as qml
from pennylane import numpy as np
dev = qml.device('default.qubit', wires=1)
def S(x):
    """Data-encoding circuit block."""
    for i in range(len(x)-1): #WE CHANGED THIS WATCH OUT
        qml.RX(x[i], wires=0)

def W(theta):
    """Trainable circuit block."""
    qml.RX(theta[0], wires=0)
    qml.RY(theta[1], wires=0)
    qml.RZ(theta[2], wires=0)
 
@qml.qnode(dev, interface="autograd")
def serial_model(weights, x):
    qml.Hadamard(wires=0)
    for q in weights:
        for theta in q[:-1]:  #FOR THETA IN ALL THE WEIGHTS UP TO (BUT NOT INCLUDING) THE LAST ONE 
            W(theta)
            S(x)
        # (L+1)'th unitary.
        W(q[-1]) #Now this is the last W gate as the input is the last element in the weight array.
    return qml.expval(qml.PauliZ(wires=0)) #Expectation value of the what??
