# Cell 1: Setup
from graphviz import Digraph
from IPython.display import Image
from micrograd.simple_neuron.engine import Value


# Cell 2: Computation
a = Value(-4.0)
a.label = 'a'
b = Value(2.0)
b.label = 'b'

c = a + b; c.label = 'c1'
d = a * b + b**3; d.label = 'd1'

c = c + c + 1; c.label = 'c2'
c = c + 1 + c + (-a); c.label = 'c3'

d = d + d * 2 + (b + a).relu(); d.label = 'd2'
d = d + 3 * d + (b - a).relu(); d.label = 'd3'

e = c - d; e.label = 'e'
f = e**2; f.label = 'f'
g = f / 2.0; g.label = 'g1'
g = g + 10.0 / f; g.label = 'g'

g.backward()


# Cell 3: Visualization
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        label = getattr(n, 'label', '')
        dot.node(name=uid, label=f"{label} | data={n.data:.4f} | grad={n.grad:.4f}", shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

# Generate and display graph inline
dot = draw_dot(g)
output_path = dot.render("micrograd_computation_graph", format="png", view=False)
Image(filename=output_path, width=1400)
