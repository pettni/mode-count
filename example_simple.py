"""
Example illustrating mode-counting synthesis on a manually defined graph.
"""

import networkx as nx
import matplotlib.pyplot as plt 

from modecount import *

# Define a simple graph
G = nx.DiGraph()
G.add_nodes_from([1,2,3,4,5,6,7,8])

G.add_path([6,5,2,1,1], mode=2)
G.add_path([8,7,4], mode=2)
G.add_path([4,3,2], mode=2)
G.add_path([1,2,3,6], mode=1)
G.add_path([5,4,6] ,mode=1)
G.add_path([6,7,8,8], mode=1)

# Set up the mode-counting problem
problem_data = {}

# Graph
problem_data['graph'] = G

# Specify initial system distribution (sums to 30)
problem_data['init'] = [0, 1, 6, 4, 7, 10, 2, 0] 

problem_data['horizon'] = 5
problem_data['cycle_set'] = [ cycle for cycle in nx.simple_cycles(G) ]  # all simple cycles (works because G is small)

# We want to bound mode-1-count between 15 and 16
problem_data['mode'] = 1
problem_data['lb'] = 15
problem_data['ub'] = 16

# Optional arguments
problem_data['lb_prefix'] = 15
problem_data['ub_prefix'] = 16
problem_data['order_function'] = G.nodes().index
problem_data['forbidden_nodes'] = G.nodes_with_selfloops()
problem_data['ilp'] = True

# mode-counting synthesis
solution_data = prefix_suffix_feasible(problem_data, verbosity = 1)