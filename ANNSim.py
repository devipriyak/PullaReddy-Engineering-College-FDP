import numpy as np

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Softmax activation
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    return exp_z / np.sum(exp_z)

# Input vector
x = np.array([1.0, 2.0, 3.0])

# Hidden layer weights (4 neurons x 3 inputs)
W_hidden = np.array([
    [0.2, 0.4, 0.6],
    [0.1, 0.3, 0.5],
    [0.3, 0.2, 0.1],
    [0.5, 0.6, 0.7]
])

# Hidden layer biases (4 neurons)
b_hidden = np.array([0.5, 0.4, 0.2, 0.3])

# Compute hidden layer linear combination
z_hidden = np.dot(W_hidden, x) + b_hidden
# Apply ReLU activation
a_hidden = relu(z_hidden)

print("Hidden layer output:", a_hidden)

# Output layer weights (2 outputs x 4 hidden neurons)
W_output = np.array([
    [0.3, 0.2, 0.4, 0.6],
    [0.5, 0.1, 0.3, 0.2]
])

# Output layer biases
b_output = np.array([0.1, 0.2])

# Compute output layer
z_output = np.dot(W_output, a_hidden) + b_output
# Apply Softmax
output = softmax(z_output)

print("Output probabilities:", output)
