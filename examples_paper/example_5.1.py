"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""

import networkx as nx

from numpy.linalg import norm
from scipy.linalg import expm

import sys
import cPickle as pickle

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
eta = 0.05		 # space discretization
 
tau = 0.5		 # time discretization

# Verify that abstraction is 0.15-approximate bisimulation
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

# randomize an initial condition
init = np.zeros(len(G))
np.random.seed(0)
j = 0
while j < 10000:
	i = np.random.randint( len(G), size=1)
	if ab.idx_to_node(i) not in forbidden_nodes:
		init[i] += 1
		j += 1


# some random cycles
cycle_set = []
c_quot_set = set([])
while len(cycle_set) < 100:
	c = random_cycle(G, 5, 0.8)
	c_quot = cyclequot(G,c,1)
	if c_quot not in c_quot_set:
		cycle_set.append(c)
		c_quot_set.add(c_quot)

# mode counting synthesis parameters
prob_data = {}
prob_data['graph'] = ab.graph
prob_data['init'] = init
prob_data['horizon'] = 10
prob_data['mode'] = 1
prob_data['lb_suffix'] = 7000
prob_data['ub_suffix'] = 7000

prob_data['cycle_set'] = cycle_set

prob_data['order_function'] = order_fcn
prob_data['forbidden_nodes'] = forbidden_nodes
prob_data['ilp'] = True

sol_data = prefix_suffix_feasible(prob_data, verbosity = 2, solver='gurobi')

pickle.dump((G, sol_data), open('example_5.1.save', 'wb') )
