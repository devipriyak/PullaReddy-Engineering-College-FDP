import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Dummy dataset with 3 input features (for demo purposes)
# Let's say we have 100 samples with 3 features each
X = np.random.rand(100, 3)
y = np.random.randint(0, 2, size=(100,))  # Binary classification (0 or 1)

# Convert labels to one-hot (if you want softmax output with 2 neurons)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=2)

#This creates a linear stack of layers (i.e., one layer after another).
#building a feedforward ANN (Dense network).
# Define the ANN model
model = Sequential()
# Input layer is defined with input_shape=(3,)
# 4 hidden layers with ReLU activation
#Dense(64)	Fully connected layer with 64 neurons (i.e., 64 hidden units).
model.add(Dense(64, input_shape=(3,), activation='relu'))  # Hidden Layer 1
model.add(Dense(32, activation='relu'))                    # Hidden Layer 2
model.add(Dense(16, activation='relu'))                    # Hidden Layer 3
model.add(Dense(8, activation='relu'))                     # Hidden Layer 4
#Reducing Neurons - is called progressive compression, often used in deep models.
# Output layer (2 neurons for 2-class classification, softmax)
model.add(Dense(2, activation='softmax'))
# Compile the model
#adam: Optimizer that adjusts weights using gradients.


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()

# Train the model (quick demo)
model.fit(X, y, epochs=10, batch_size=8, verbose=1)
