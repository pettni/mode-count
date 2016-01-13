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

g = 1 # gravitational coefficient 	m/s^2
k = 1  # spring coefficient 		N/m = kg / s^2
m = 0.1 # pendulum mass				kg
l = 1  # pendulum arm length 		m

umax = 0.5

# Define a vector fields
vf1 = lambda x : [x[1], -(g/l) * np.sin(x[0]) - (k/m) * np.sin(x[1]) + umax]
vf2 = lambda x : [x[1], -(g/l) * np.sin(x[0]) - (k/m) * np.sin(x[1]) - umax]
     
# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
omega = k/m 
lambda1 = (omega**2 + 2 - np.sqrt(omega**4 + 4))/4
lambda2 = (omega**2 + 2 + np.sqrt(omega**4 + 4))/4

zeta_min = 0.84
zeta_max = 1

alpha1 = lambda1/2
alpha2 = lambda2/2
rho = 0.5* (omega * np.min([(g/l)*zeta_min, 1]) - (g/l) * zeta_max )
sigma = omega + 2

if rho < 0:
	print 'rho negative:', rho
	assert(False)

kl1 = lambda r,s : np.sqrt(alpha2/alpha1) * np.exp( -(rho/2*alpha2) * s) * r
kl2 = lambda r,s : np.sqrt(alpha2/alpha1) * np.exp( -(rho/2*alpha2) * s) * r

# Define an abstraction
data = {}


# Abstraction parameters
lb = [-1, -1] # lower bounds
ub = [1, 1]		 # upper bounds
eta = 0.1		 # space discretization
 
tau = 0.5		 # time discretization

epsilon = 0.05

tvec = np.arange(0,1,0.01)
plt.plot(tvec, kl1(epsilon, tvec))
plt.plot(tvec, [epsilon-eta/2] * len(tvec))
plt.show()

# Verify that abstraction is 0.1-approximate bisimulation
# with respect to both KL functions
# assert(verify_bisim( kl1, tau, eta, epsilon))
# assert(verify_bisim( kl2, tau, eta, epsilon))

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
mc_sol = synthesize_feas(G, init, T, mode_des, mode_des, mode, cycle_set = cycle_set,
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

nx.draw_networkx_edges(H, pos, edgelist = edges(mc_sol['cycles'][0]) , edge_color = 'red', style='solid', width=1.5)
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

