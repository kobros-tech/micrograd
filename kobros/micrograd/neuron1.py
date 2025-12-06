from simple_neuron.engine import Value
from simple_neuron.nn import MLP

model = MLP(3, [4, 4, 1])  # 3 inputs → 2 hidden layers → 1 output
x = [1.0, 2.0, -1.0]       # inputs must be floats
y = Value(1.0)             # target output

out = model(x)
loss = (out - y) ** 2
loss.backward()

print("Loss:", loss.data)
print("Gradient of first weight:", model.parameters()[0].grad)
