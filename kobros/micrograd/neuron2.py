import torch

# Step 1: Define model (3 inputs → 2 hidden layers → 1 output)
model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),   # first hidden layer (3 → 4)
    torch.nn.ReLU(),
    torch.nn.Linear(4, 4),   # second hidden layer (4 → 4)
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1)    # output layer (4 → 1)
)

# Step 2: Input and target
x = torch.tensor([1.0, 2.0, -1.0])      # input vector
y = torch.tensor([1.0])                # target output

# Step 3: Forward pass
out = model(x)
loss = (out - y) ** 2  # MSE loss

# Step 4: Backward pass
loss.backward()

# Step 5: Print loss and gradient of first weight
print("Loss:", loss.item())
print("Gradient of first weight:", model[0].weight.grad)


# Inputs with gradient tracking
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
x3 = torch.tensor(-1.0, requires_grad=True)

# Define function directly
y = 2 * x1 - 3 * x2 + x3 + 1

# Backward pass
y.backward()

# Print exact partial derivative ∂y/∂x1
print("∂y/∂x1 =", x1.grad.item())  # Should print 2.0
print("∂y/∂x2 =", x2.grad.item())  # Should print -3.0
print("∂y/∂x3 =", x3.grad.item())  # Should print 1.0
