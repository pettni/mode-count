"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""

import networkx as nx
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.linalg import expm

from modecount import *
from make_integer import make_integer

# Define an abstraction
data = {}
# Define a vector fields
vf1 = lambda x : [-(x[0]-1.05) + x[1], -(x[0]-1.05) - x[1]]
vf2 = lambda x : [-(x[0]+1.05) + x[1], -(x[0]+1.05) - x[1]]
              
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

# extract abstraction graph
G = ab.graph

# randomize an initial condition
init = np.zeros(len(G))
np.random.seed(1)
for i in np.random.randint( len(G), size=50):
	init[i] += 1

# mode counting synthesis parameters
T = 10 			# horizon
mode_des = 28	# desired mode count over time
mode = 1		# mode to count (1 or 2)
forbidden_nodes = G.nodes_with_selfloops()

# order fcn def
order_fcn = ab.node_to_idx

# mode-counting synthesis
mc_sol = synthesize(G, init, T, mode_des, mode, 
			forbidden_nodes = forbidden_nodes, integer = False, order_fcn = order_fcn, 
			verbosity = 1)

nonint_cycles = mc_sol['cycles']
nonint_assignments = mc_sol['assignments']
int_assignments = make_integer(nonint_assignments)

print cycles_maxmin(G,nonint_cycles, mode, nonint_assignments)
print cycles_maxmin(G,nonint_cycles, mode, int_assignments)

mc_sol2 = reach_cycles(G, init, T, mode, nonint_cycles, int_assignments, forbidden_nodes = forbidden_nodes, integer = False, order_fcn = order_fcn, 
			verbosity = 1)


# simulate it on the connected subset of the graph!
strongly_conn_nodes = G.subgraph(max(nx.strongly_connected_components(G), key=len))
anim = simulate(G, mc_sol, order_fcn, strongly_conn_nodes)
anim.save('example_abstraction_lin2d_anim.mp4', fps=10)

plt.show()
