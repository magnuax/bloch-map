import numpy as np
import utils as utils

def depolarizing(X, p=0.5):
    out = (1-p)*X + p*np.eye(2)/2
    return out

def phase_damping(X, p=0.5):
    out = (1-p)*X + p*utils.pauli_z*X*utils.pauli_z
    return out

def amplitude_damping(X, p=0.5):
    K0 = np.array([[1, 0], [0, np.sqrt(1-p)]])
    K1 = np.array([[0, np.sqrt(p)], [0, 0]])
    out = np.dot(K0, np.dot(X, K0.T)) + np.dot(K1, np.dot(X, K1.T))
    return out

def holevo_werner(X, p=0.5):
    out = 0.5*(np.trace(X)*np.eye(2) - X.T)
    return out

def resonant_amplitude_damping(X, gamma_10=0.5):
    gamma_11 = (1 - gamma_10)/2
    gamma_00 = (1 - gamma_10)/2
    
    K_0 = np.sqrt(gamma_00)*np.array([[1, 0], [0, 0]]) + np.sqrt(gamma_11)*np.array([[0, 0], [0, 1]])
    K_1 = np.sqrt(gamma_10)*np.array([[0, 1], [0, 0]])
    
    out = np.dot(K_0, np.dot(X, K_0.T)) + np.dot(K_1, np.dot(X, K_1.T))
    
    return out
