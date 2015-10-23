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
from make_integer import make_integer


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
 
tau = 1.		 # time discretization

# Verify that abstraction is 0.3-approximate bisimulation
# with respect to both KL functions
verify_bisim( kl1, tau, eta, 0.1 )
verify_bisim( kl2, tau, eta, 0.1 )

# Initiate abstraction
ab = Abstraction(lb, ub, eta, tau)


# add modes to abstraction
ab.add_mode(vf1)
ab.add_mode(vf2)

# extract abstraction graph
G = ab.graph
print "abstraction has ", len(G), " states"

# order fcn def
order_fcn = ab.node_to_idx
forbidden_nodes = G.nodes_with_selfloops()

# randomize an initial condition
init = np.zeros(len(G))
np.random.seed(1)
j = 0
while j < 10000:
	i = np.random.randint( len(G), size=1)
	if ab.idx_to_node(i) not in forbidden_nodes:
		init[i] += 1
		j += 1

# mode counting synthesis parameters
T = 10 			# horizon
mode_des = 7000 # desired mode count over time
mode = 1		# mode to count (1 or 2)

# mode-counting synthesis
mc_sol = synthesize(G, init, T, mode_des, mode, 
			forbidden_nodes = forbidden_nodes, integer = True, order_fcn = order_fcn, 
			verbosity = 2)

H = max(nx.strongly_connected_component_subgraphs(G), key=len)

edgelist1 = [(u,v) for (u,v,d) in H.edges(data=True) if d['mode'] == 1 ]
edgelist2 = [(u,v) for (u,v,d) in H.edges(data=True) if d['mode'] == 2 ]

pos = nx.spring_layout(H)

nx.draw_networkx_nodes(H, pos, node_color = 'black', alpha=0.5)
nx.draw_networkx_edges(H, pos, edgelist = edgelist1, edge_color = 'red', alpha=0.2)
nx.draw_networkx_edges(H, pos, edgelist = edgelist2, edge_color = 'blue', alpha=0.2)

colors = [ 'purple', 'green' ]

def edges(cycle):
	return [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]

nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][0]) , edge_color = 'purple', style='dashed', width=3.)
nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][1]) , edge_color = 'green', style='solid', width=1.5)

xmin = min([val[0] for val in pos.itervalues()]) - 0.05
xmax = max([val[0] for val in pos.itervalues()]) + 0.05
ymin = min([val[1] for val in pos.itervalues()]) - 0.05
ymax = max([val[1] for val in pos.itervalues()]) + 0.05

#Options
params = {'figure.figsize': (3,2),
          }
plt.axis('off')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.rcParams.update(params) 
plt.savefig('ex2d_graph.pdf', format='pdf')

