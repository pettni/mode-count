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

# Plot it
# draw_modes(G)
# plt.show()

# Specify initial system distribution (sums to 30)
init = [0, 1, 6, 4, 7, 10, 2, 0]

T = 5 			# horizon
mode_des = 15	# desired mode count over time
mode = 1		# mode to count (1 or 2)

forbidden_nodes = G.nodes_with_selfloops()

# mode-counting synthesis
mc_sol = synthesize(G, init, T, mode_des, mode, forbidden_nodes = forbidden_nodes, integer = False, verbosity = 1)

nonint_cycles = mc_sol['cycles']
nonint_assignments = mc_sol['assignments']
int_assignments = make_integer(nonint_assignments)

print mc_sol['controls']

print cycles_maxmin(G,nonint_cycles, mode, nonint_assignments)
print cycles_maxmin(G,nonint_cycles, mode, int_assignments)

mc_sol2 = reach_cycles(G, init, T, mode, nonint_cycles, int_assignments, forbidden_nodes = forbidden_nodes, integer = False, 
			verbosity = 1)

print mc_sol2['controls']

# # simulate it on the graph!
# anim = simulate(G, mc_sol, lambda node : G.nodes().index(node))
# anim.save("example_simple_anim.mp4", fps=10)
# plt.show()