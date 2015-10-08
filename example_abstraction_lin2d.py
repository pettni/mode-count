"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""

import networkx as nx
import matplotlib.pyplot as plt

from treeabstr import *

# Define an abstraction
data = {}
# Define a vector fields
vf1 = lambda x : [-(x[0]-1) + x[1], -(x[0]-1) - x[1]]
vf2 = lambda x : [-(x[0]+1) + x[1], -(x[0]+1) - x[1]]
              
# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
kl1 = lambda r,s : r * norm( expm(s*np.array([[-1,  1], [-1, -1]])) , np.inf)
kl2 = lambda r,s : r * norm( expm(s*np.array([[-1, -1], [ 1, -1]])) , np.inf)

# Abstraction parameters
lb = [-2, -1]	# lower bounds
ub = [2, 1]		# upper bounds
eta = 0.1		# space discretization

tau = 1.1		# time discretization
eps = 0.3		# desired bisimulation approximation

# Initiate abstraction
ab = Abstraction(lb, ub, eta, tau, eps)

# Verify that abstraction is eps-approximate bisimulation
# with respect to both KL functions
assert(ab.verify_bisim(kl1))
assert(ab.verify_bisim(kl2))

# add modes to abstraction
ab.add_mode(vf1, tau)
ab.add_mode(vf2, tau)

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

# mode-counting synthesis
mc_sol = synthesize(G, init, T, mode_des, mode, verbosity = 1)

# simulate it on the connected subset of the graph!
strongly_conn_nodes = G.subgraph(max(nx.strongly_connected_components(G), key=len))
anim = simulate(G, mc_sol, strongly_conn_nodes)
anim.save('example_abstraction_lin2d_anim.mp4', fps=10)

plt.show()
