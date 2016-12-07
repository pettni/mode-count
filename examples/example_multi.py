"""
Example illustrating mode-counting synthesis on a manually defined graph.
"""
import sys
sys.path.append('../')

import networkx as nx
from itertools import product
from copy import deepcopy

from modecount_new import *


# Define a simple graph
G1 = ModeGraph()
G1.add_nodes_from([1, 2, 3])

G1.add_path([1, 3, 3], modes=[1])
G1.add_path([1, 2], modes=[0])
G1.add_path([2, 2], modes=[1])

G2 = deepcopy(G1)

# Set up the mode-counting problem
cp = MultiCountingProblem()

cp.graphs.append(G1)
cp.graphs.append(G2)

cp.constraints.append(
	([[(2,0), (2,1)], set()], 0)
)
cp.constraints.append(
	([set(), [(3,0), (3,1)]], 0)
)

cp.inits.append([4, 0, 0])
cp.inits.append([4, 0, 0])

cp.T = 3
cp.cycle_sets.append([[(3,1)], [(2,1)]])
cp.cycle_sets.append([[(3,1)], [(2,1)]])

cp.solve_prefix_suffix()

cp.test_solution()

# print cp.x
# print cp.u
# print cp.assignments