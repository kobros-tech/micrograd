from simple_neuron.engine import Value
from simple_neuron.nn import MLP

model = MLP(3, [4, 4, 1])  # 3 inputs → 2 hidden layers → 1 output

x = [1.0, -2.0, 3.0]  # input vector
y = Value(1.0)  # target

out = model(x)        # forward pass
loss = (out - y)**2   # mean squared error
loss.backward()       # backward pass

for p in model.parameters():
    p.data += -0.01 * p.grad  # manual gradient descent step
