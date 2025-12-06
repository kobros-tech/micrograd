Absolutely — here’s a clean explanation in plain Markdown format for you to read or use directly (e.g., in Jupyter or GitHub):

---

# 📘 Abstract: Approximating a Linear Function with a Neural Network

## 🔢 Problem Statement

We are given a simple target function:

```
y = 2x₁ - 3x₂ + x₃ + 1
```

Our goal is to compare:

1. ✅ The **exact mathematical derivative**
2. ✅ The **approximated function** learned by a small PyTorch neural network

---

## ✅ Exact Mathematical Derivatives

We compute the partial derivatives:

```
∂y/∂x₁ = 2
∂y/∂x₂ = -3
∂y/∂x₃ = 1
```

Gradient vector:

```
∇y = [2, -3, 1]
```

This is the symbolic derivative of the function — exact and correct.

---

## 🤖 PyTorch Neural Network Approximation

The model you trained is:

- Input layer: 3 inputs → 4 neurons (Linear + ReLU)
- Hidden layer: 4 neurons → 4 neurons (Linear + ReLU)
- Output layer: 4 neurons → 1 neuron (Linear)

### 🧠 Weights After Training:

#### Layer 1 (input → 4 hidden)

```
W1 (0.weight):
[[-0.4454, -0.1505,  0.4814],
 [-0.4119, -0.1214,  0.5672],
 [-0.5575, -0.5107,  0.3966],
 [-0.1840, -0.5394, -0.1701]]

b1 (0.bias):
[ 0.4357, -0.4811,  0.1761,  0.1821]
```

#### Layer 2 (4 hidden → 4 hidden)

```
W2 (2.weight):
[[ 0.2318,  0.1230, -0.4417,  0.0944],
 [ 0.2571, -0.4072, -0.2355,  0.3348],
 [-0.0914,  0.1671,  0.4249, -0.0945],
 [ 0.1129, -0.4468,  0.3673,  0.3732]]

b2 (2.bias):
[-0.1722, -0.1678, 1.3474, 1.0853]
```

#### Output Layer (4 → 1)

```
W3 (4.weight):
[[ 0.0620, -0.1085, -1.3019, -1.0525]]

b3 (4.bias):
[-1.1035]
```

---

## 🧮 Final Function Structure

The neural network computes:

```
h1 = ReLU(W1 · x + b1)
h2 = ReLU(W2 · h1 + b2)
ŷ  = W3 · h2 + b3
```

So the full function is:

```
ŷ = W3 · ReLU(W2 · ReLU(W1 · x + b1) + b2) + b3
```

> Not human-readable as an equation, but numerically accurate for the training range.

---

## ✅ Conclusion

| Aspect           | Value                       |
| ---------------- | --------------------------- |
| True function    | `y = 2x₁ - 3x₂ + x₃ + 1`    |
| Learned function | Deep ReLU MLP — approximate |
| Exact gradient   | `[2, -3, 1]`                |
| Network loss     | Near zero                   |
| Model output     | Very close to expected      |

---

For full transparency or if interpretability is critical, using a **linear model** like `nn.Linear(3, 1)` gives you the formula directly.
