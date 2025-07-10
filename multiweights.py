w = 0.0
for epoch in range(10):
    y_pred = w * x
    loss = (y_pred - y)**2
    grad = 2 * (y_pred - y) * x
    w = w - lr * grad
    print(f"Epoch {epoch+1}: w = {w:.4f}, loss = {loss:.4f}, grad = {grad:.4f}")
