# Input and true output
x = 2
y = 4

# Initialize weight
w = 0.0

# Learning rate
lr = 0.1

# Training loop (just 1 epoch for now)
for epoch in range(1):
    # Prediction
    y_pred = w * x

    # Loss
    loss = (y_pred - y)**2

    # Gradient computation
    grad = 2 * (y_pred - y) * x

    # Update weight
    w = w - lr * grad

    # Output values
    print(f"Epoch {epoch+1}: w = {w:.4f}, loss = {loss:.4f}, grad = {grad:.4f}")
