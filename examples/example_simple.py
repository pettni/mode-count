"""
Example illustrating mode-counting synthesis on a manually defined graph.
"""
import sys

import networkx as nx
import matplotlib.pyplot as plt 
from itertools import product

sys.path.append('../')
from counting import *

# Define a simple graph
G = ModeGraph()
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])

G.add_path([6, 5, 2, 1, 1], modes=[2])
G.add_path([8, 7, 4], modes=[2])
G.add_path([4, 3, 2], modes=[2])
G.add_path([1, 2, 3, 6], modes=[1])
G.add_path([5, 4, 6], modes=[1])
G.add_path([6, 7, 8, 8], modes=[1])

# Set up the mode-counting problem
cp = MultiCountingProblem(1)

# Graph
cp.graphs[0] = G

# Specify initial system distribution (sums to 30)
cp.inits[0] = [0, 1, 6, 4, 7, 10, 2, 0]

cp.T = 5


# all simple cycles (works because G is small)
def augment(G, c):
    outg = [G[c[i]][c[(i + 1) % len(c)]]['modes'][0]
            for i in range(len(c))]
    return zip(c, outg)


cp.cycle_sets[0] = [augment(G, cycle) for cycle
                    in nx.simple_cycles(nx.DiGraph(G))]

# We want to bound mode-1-count between 15 and 16
cc1 = CountingConstraint(1)
cc1.X[0] = set(product(G.nodes(), [1]))
cc1.R = 16  # At most 16 in mode 1

cc2 = CountingConstraint(1)
cc2.X[0] = set(product(G.nodes(), [2]))
cc2.R = 30 - 15  # At least 15 in mode 1

cc3 = CountingConstraint(1)
cc3.X[0] = set(product(G.nodes_with_selfloops(), [1, 2]))
cc3.R = 0  # Forbid nodes with self loops

cp.constraints += [cc1, cc2, cc3]

cp.solve_prefix_suffix()

cp.test_solution()
