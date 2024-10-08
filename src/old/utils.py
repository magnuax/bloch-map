import numpy as np

# Define Pauli matrices:
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1,  0], [0, -1]])
def state_to_bloch(state):
    """
    Converts qubit state (array, 2x2) to corresponding Bloch vector (array, 3).
    """
    x = np.trace(np.dot(state, pauli_x)).real
    y = np.trace(np.dot(state, pauli_y)).real
    z = np.trace(np.dot(state, pauli_z)).real
    bloch_vector = np.array([x, y, z])

    return bloch_vector

def bloch_to_state(bloch_vector):
    """
    Converts a Bloch vector (array, 3) to corresponding qubit state (array, 2x2).
    """
    x, y, z = bloch_vector
    state = 0.5*(np.eye(2) + x*pauli_x + y*pauli_y + z*pauli_z)
    return state

def example_map(X):
    x1, x2, x3, x4 = X.flatten()

    out = np.array([[1/3*x1 + 2/3*x4, 1/6*x2],
                    [1/6*x3, 1/3*x4 + 2/3*x1]])
    return out    

def depolarizing_channel(X, p=0.5):
    out = (1-p)*X + p*np.eye(2)/2
    return out

def dephasing_x_channel(X, p=0.5):
    out = (1-p)*X + p*pauli_x*X*pauli_x
    return out

def dephasing_y_channel(X, p=0.5):
    out = (1-p)*X + p*pauli_y*X*pauli_y
    return out

def dephasing_z_channel(X, p=0.5):
    out = (1-p)*X + p*pauli_z*X*pauli_z
    return out

def phase_damping_channel(X, p=0.5):
    out = (1-p)*X + p*pauli_z*X*pauli_z
    return out

def amplitude_damping_channel(X, p=0.5):
    K0 = np.array([[1, 0], [0, np.sqrt(1-p)]])
    K1 = np.array([[0, np.sqrt(p)], [0, 0]])
    out = np.dot(K0, np.dot(X, K0.T)) + np.dot(K1, np.dot(X, K1.T))

    return out

def holevo_werner_channel(X, p=0.5):
    out = 0.5*(np.trace(X)*np.eye(2) - X.T)
    return out

def resonant_amplitude_damping_channel(X, gamma_10=0.5):
    gamma_11 = (1 - gamma_10)/2
    gamma_00 = (1 - gamma_10)/2
    
    K_0 = np.sqrt(gamma_00)*np.array([[1, 0], [0, 0]]) + np.sqrt(gamma_11)*np.array([[0, 0], [0, 1]])
    K_1 = np.sqrt(gamma_10)*np.array([[0, 1], [0, 0]])
    
    out = np.dot(K_0, np.dot(X, K_0.T)) + np.dot(K_1, np.dot(X, K_1.T))
    
    return out
