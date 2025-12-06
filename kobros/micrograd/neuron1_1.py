from simple_neuron.engine import Value
from simple_neuron.nn import MLP
import random

# Example:
# y = 2 * x1 - 3 * x2 + x3 + 1

def mse_loss(pred, target):
    """Mean squared error loss"""
    return (pred - target) ** 2

# Initialize model: 3 inputs → 2 hidden layers → 1 output
model = MLP(3, [4, 4, 1])

# Training loop
epochs = 100
learning_rate = 0.01
samples_per_epoch = 10

for epoch in range(epochs):
    total_loss = 0

    for _ in range(samples_per_epoch):
        # Generate synthetic input and expected output
        x_raw = [random.uniform(-1, 1) for _ in range(3)]
        y_true = 2 * x_raw[0] - 3 * x_raw[1] + x_raw[2] + 1  # True function

        # Forward pass
        x = x_raw                      # raw floats for input
        y = Value(y_true)             # wrap target in Value
        out = model(x)                # model output
        loss = mse_loss(out, y)       # compute loss
        total_loss += loss.data       # accumulate loss for logging

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Gradient descent step
        for p in model.parameters():
            p.data += -learning_rate * p.grad

    # End of epoch: report loss
    print(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")

#  After training
x_test = [1.0, 2.0, -1.0]
out = model(x_test)
print("Loss:", out.data)
print("Gradient of first weight:", model.parameters()[0].grad)
