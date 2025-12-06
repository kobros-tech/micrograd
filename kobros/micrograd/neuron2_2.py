import torch
import torch.nn as nn
import torch.nn.functional as F

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def true_function(x):
    return 2 * x[0] - 3 * x[1] + x[2] + 1

for epoch in range(100):
    optimizer.zero_grad()
    x = torch.tensor([1.0, 2.0, -1.0])
    y_true = torch.tensor([true_function(x)])
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    if epoch >= 97:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 3: Manual test
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
x3 = torch.tensor(-1.0, requires_grad=True)
y = 2 * x1 - 3 * x2 + x3 + 1
y.backward()
print(f"\n📘 Exact derivatives:")
print(f"∂y/∂x1 = {x1.grad.item()}")
print(f"∂y/∂x2 = {x2.grad.item()}")
print(f"∂y/∂x3 = {x3.grad.item()}")

# Step 4: Print learned parameters cleanly
print("\n📦 Learned parameters (PyTorch):")
for name, param in model.named_parameters():
    print(f"{name}:\n{param.data}\n")