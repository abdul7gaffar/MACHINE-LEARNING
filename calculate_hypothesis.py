import numpy as np
from sigmoid import *

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0.0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    theta_t=np.transpose(theta)
    hypothesis=np.dot(theta_t,X[i])
    ########################################/

    return sigmoid(hypothesis)

