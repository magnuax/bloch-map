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

def get_sphere_mesh(N):
    """
    Generates NxN mesh of points on the unit sphere.
    """
    theta = np.linspace(0, 2*np.pi, N)
    phi   = np.linspace(0, np.pi, N)    
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    
    return x,y,z