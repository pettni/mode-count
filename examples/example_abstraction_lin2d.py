"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""
import networkx as nx
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.linalg import expm
from itertools import product

import sys
sys.path.append('../')
from abstraction import *
from counting import *

# Define a vector fields
vf1 = lambda x: [-(x[0] - 1.0) + x[1], -(x[0] - 1.0) - x[1]]
vf2 = lambda x: [-(x[0] + 1.0) + x[1], -(x[0] + 1.0) - x[1]]

# Abstraction parameters
lb = [-2, -1.05]  # lower bounds
ub = [2, 1.05]    # upper bounds
eta = 0.1        # space discretization

tau = 1.1        # time discretization

# Initiate abstraction
ab = Abstraction(lb, ub, eta, tau)

# Verify that abstraction is 0.3-approximate bisimulation
# with respect to both KL functions
verify_bisim(lambda r, s: r * norm(expm(s * np.array([[-1, 1],
                                                      [-1, -1]])),
                                   np.inf),
             tau, eta, 0.3, 0, 1)

# add modes to abstraction
ab.add_mode(vf1)
ab.add_mode(vf2)

# randomize an initial condition
init = np.zeros(len(ab.graph))
np.random.seed(1)
for i in np.random.randint(len(ab.graph), size=50):
    init[i] += 1

# Define discrete mode-count synthesis problem
cp = MultiCountingProblem(1)

cp.graphs[0] = ab.graph
cp.inits[0] = init
cp.cycle_sets[0] = [augment(ab.graph, c) for c
                    in nx.simple_cycles(nx.DiGraph(ab.graph))]

cp.T = 10

cc1 = CountingConstraint(1)
cc1.X[0] = set(product(ab.graph.nodes(), [1]))
cc1.R = 33

cc2 = CountingConstraint(1)
cc2.X[0] = set(product(ab.graph.nodes(), [2]))
cc2.R = 50 - 33

cp.constraints += [cc1, cc2]

cp.solve_prefix_suffix()
cp.test_solution()
