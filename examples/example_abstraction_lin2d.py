"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""

import networkx as nx
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.linalg import expm

import sys
sys.path.append('../')
from modecount import *

# Define an abstraction
data = {}
# Define a vector fields
vf1 = lambda x : [-(x[0]-1.0) + x[1], -(x[0]-1.0) - x[1]]
vf2 = lambda x : [-(x[0]+1.0) + x[1], -(x[0]+1.0) - x[1]]
              
# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
kl1 = lambda r,s : r * norm( expm(s*np.array([[-1,  1], [-1, -1]])) , np.inf)
kl2 = lambda r,s : r * norm( expm(s*np.array([[-1, -1], [ 1, -1]])) , np.inf)

# Abstraction parameters
lb = [-2, -1.05] # lower bounds
ub = [2, 1.05]		 # upper bounds
eta = 0.1		 # space discretization
 
tau = 1.1		 # time discretization

# Initiate abstraction
ab = Abstraction(lb, ub, eta, tau)

# Verify that abstraction is 0.3-approximate bisimulation
# with respect to both KL functions
verify_bisim( kl1, tau, eta, 0.3 )
verify_bisim( kl2, tau, eta, 0.3 )

# add modes to abstraction
ab.add_mode(vf1)
ab.add_mode(vf2)

# randomize an initial condition
init = np.zeros(len(ab.graph))
np.random.seed(1)
for i in np.random.randint( len(ab.graph), size=50):
	init[i] += 1

# Define discrete mode-count synthesis problem
pre_suf_data = {}
pre_suf_data['graph'] = ab.graph			 # graph
pre_suf_data['order_function'] = ab.node_to_idx   # function mapping node -> integer (improves speed if defined)

pre_suf_data['init'] = init 					# initial state configuration
pre_suf_data['forbidden_nodes'] = ab.graph.nodes_with_selfloops() 	# nodes that can't be visited

pre_suf_data['mode'] = 1 					# mode to control
pre_suf_data['lb_suffix'] = 33 					# lower mode-count bound
pre_suf_data['ub_suffix'] = 33 					# upper mode-count bound

pre_suf_data['ilp'] = True

pre_suf_data['horizon'] = 10 			 	# prefix horizon

pre_suf_data['cycle_set'] = list(nx.simple_cycles(ab.graph))

# Solve discrete mode synthesis problem
pre_suf_sol = prefix_suffix_feasible(pre_suf_data, verbosity=2)

# simulate it on the connected subset of the graph!
G = ab.graph
strongly_conn_nodes = G.subgraph(max(nx.strongly_connected_components(G), key=len))
anim = simulate(G, pre_suf_sol, ab.node_to_idx, strongly_conn_nodes)
# anim.save('example_abstraction_lin2d_anim.mp4', fps=10)

plt.show()
