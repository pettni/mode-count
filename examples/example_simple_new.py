"""
Example illustrating mode-counting synthesis on a manually defined graph.
"""
import sys
sys.path.append('../')

import networkx as nx
from itertools import product
from modecount_new import *


# Define a simple graph
G = nx.DiGraph()
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])

G.add_path([6, 5, 2, 1, 1], modes=[1])
G.add_path([8, 7, 4], modes=[1])
G.add_path([4, 3, 2], modes=[1])
G.add_path([1, 2, 3, 6], modes=[0])
G.add_path([5, 4, 6], modes=[0])
G.add_path([6, 7, 8, 8], modes=[0])

# Set up the mode-counting problem
cp = SingleCountingProblem(G)

cp.add_constraint(list(product(G.nodes(), [0])), 16)       			  # at most 16 in 0
cp.add_constraint(list(product(G.nodes(), [1])), 30 - 15)  			  # at least 15 in 0
cp.add_constraint(list(product(G.nodes_with_selfloops(), [0,1])), 0)  # forbidden set

init = [0, 1, 6, 4, 7, 10, 2, 0]
horizon = 5

def outg(c):
	return [G[c[i]][c[(i+1) % len(c)]]['modes'][0] for i in range(len(c))]

cycle_set = [zip(c, outg(c)) for c in nx.simple_cycles(G)]

cp.solve_prefix_suffix(init, horizon, cycle_set)

cp.test_solution()

# # Specify initial system distribution (sums to 30)
# problem_data['init'] = [0, 1, 6, 4, 7, 10, 2, 0]

# problem_data['horizon'] = 5
# problem_data['cycle_set'] = [cycle for cycle in nx.simple_cycles(G)]

# # We want to bound mode-1-count between 15 and 16
# problem_data['mode'] = 1
# problem_data['lb_suffix'] = 15
# problem_data['ub_suffix'] = 16

# # Optional arguments
# problem_data['lb_prefix'] = 15
# problem_data['ub_prefix'] = 16
# problem_data['order_function'] = G.nodes().index
# problem_data['forbidden_nodes'] = G.nodes_with_selfloops()
# problem_data['ilp'] = True

# mode-counting synthesis
# solution_data = prefix_suffix_feasible(problem_data, verbosity=2)
