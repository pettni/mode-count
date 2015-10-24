"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
eta = 0.05		 # space discretization
 
tau = 0.5		 # time discretization

# Verify that abstraction is 0.1-approximate bisimulation
# with respect to both KL functions
assert(verify_bisim( kl1, tau, eta, 0.15 ))
assert(verify_bisim( kl2, tau, eta, 0.15 ))

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
forbidden_nodes = [ node for node, attr in G.nodes_iter(data=True) if np.all(np.abs(attr['mid']) < 0.3) ]
# print [G.node[node]['mid'] for node in forbidden_nodes]

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

# some random cycles
cycle_set = []
c_quot_set = set([])
while len(cycle_set) < 100:
	c = random_cycle(G, 5, 0.8)
	c_quot = cyclequot(G,c,mode)
	if c_quot not in c_quot_set:
		cycle_set.append(c)
		c_quot_set.add(c_quot)


# mode-counting synthesis
mc_sol = synthesize(G, init, T, mode_des, mode, cycle_set = cycle_set,
			forbidden_nodes = forbidden_nodes, integer = True, order_fcn = order_fcn, 
			verbosity = 2)

H = max(nx.strongly_connected_component_subgraphs(G), key=len)

edgelist1 = [(u,v) for (u,v,d) in H.edges(data=True) if d['mode'] == 1 ]
edgelist2 = [(u,v) for (u,v,d) in H.edges(data=True) if d['mode'] == 2 ]

# pos = nx.spring_layout(H)

scalefac = 10

pos = {}
for node, attr in H.nodes_iter(data=True):
	pos[node] = scalefac*attr['mid']

# nx.draw_networkx_edges(H, pos, edgelist = edgelist1, edge_color = 'red', alpha=0.2)
# nx.draw_networkx_edges(H, pos, edgelist = edgelist2, edge_color = 'blue', alpha=0.2)

def edges(cycle):
	return [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]

nx.draw_networkx_nodes(H, pos, nodelist = list(set.union(*[set(cyc) for cyc in mc_sol['cycles'] ])), node_color = 'black', alpha=0.5)

nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][0]) , edge_color = 'red', style='dashed', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][1]) , edge_color = 'green', style='solid', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][2]) , edge_color = 'blue', style='solid', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][3]) , edge_color = 'yellow', style='solid', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][4]) , edge_color = 'purple', style='solid', width=1.5)

xmin = min([val[0] for val in pos.itervalues()]) - 0.1 * scalefac
xmax = max([val[0] for val in pos.itervalues()]) + 0.1 * scalefac
ymin = min([val[1] for val in pos.itervalues()]) - 0.1 * scalefac
ymax = max([val[1] for val in pos.itervalues()]) + 0.1 * scalefac

currentAxis = plt.gca()

currentAxis.add_patch(Rectangle( (-scalefac*0.15, -scalefac*0.15), scalefac*0.3, scalefac*0.3, color='red', alpha=0.3) )
currentAxis.add_patch(Rectangle( (-scalefac*0.3, -scalefac*0.3), scalefac*0.6, scalefac*0.6, color='red', alpha=0.3) )

#Options
plt.axis('off')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig('ex2d_graph.pdf', format='pdf')

