"""
TCL example
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from modecount import *

# TCL parameters
Cth = 2.
Rth = 2.
Pm = 5.6
eta_tcl = 2.5
theta_r = 22.5
delta = 0.3

# Ambient temperature
theta_a = 32.

# Derived constants
a = 1./(Rth * Cth)
b = eta_tcl / Cth

# Define a vector fields
vf1 = lambda theta : -a * ( theta - theta_a ) - b * Pm
vf2 = lambda theta : -a * ( theta - theta_a )
              
# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
kl1 = lambda r,s : r * np.exp(-s*a)
kl2 = lambda r,s : r * np.exp(-s*a)

# Abstraction parameters
lb = [theta_r - delta]		# lower bounds
ub = [theta_r + delta]		# upper bounds
eta = 0.0005					# space discretization
tau = 0.016					# time discretization
eps = 0.1					# desired bisimulation approximation

# Initiate abstraction
ab = Abstraction(lb, ub, eta, tau)

# Verify that abstraction is eps-approximate bisimulation
# with respect to both KL functions
assert(ab.verify_bisim(kl1, eps))
assert(ab.verify_bisim(kl2, eps))
# 
# add modes to abstraction
ab.add_mode(vf1)
ab.add_mode(vf2)

# extract abstraction graph
G = ab.graph
# draw_modes(G)
# plt.show()

# randomize an initial condition
init = np.zeros(len(G))
np.random.seed(1)
for i in np.random.randint( len(G), size=10000):
	init[i] += 1

# mode counting synthesis parameters
T = 5 			# horizon
mode_des = 3000	# desired mode count over time
mode = 1		# mode to count (1 or 2)

order_fcn = ab.node_to_idx
# order_fcn = G.nodes().index

# mode-counting synthesis
mc_sol = synthesize(G, init, T, mode_des, mode, order_fcn = order_fcn, integer=False, verbosity = 1)

# simulate it on the connected subset of the graph!
strongly_conn_nodes = G.subgraph(max(nx.strongly_connected_components(G), key=len))
anim = simulate(G, mc_sol, strongly_conn_nodes)
anim.save('example_abstraction_lin2d_anim.mp4', fps=10)

plt.show()
