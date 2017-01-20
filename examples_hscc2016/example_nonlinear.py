"""
Example illustrating abstracting a 2-mode switched system, and mode-counting synthesis on the abstraction graph
"""

import cPickle as pickle
import sys
import numpy as np
sys.path.append('../')

from modecount import *
from make_integer import make_integer
from random_cycle import random_cycle

# Parameters
g = 9.8 # gravitational coefficient 	m/s^2
k = 6.  # friction coefficient 		N/m = kg / s^2
m = 0.5 # pendulum mass				kg
l = 1.  # pendulum arm length 		m

umax = 5

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
print rho

kl1 = lambda r,s : np.sqrt(alpha2/alpha1) * np.exp( -(rho/2*alpha2) * s) * r
kl2 = lambda r,s : np.sqrt(alpha2/alpha1) * np.exp( -(rho/2*alpha2) * s) * r

# Abstraction parameters
lb = [-1, -1] 	 # lower bounds
ub = [1, 1]		 # upper bounds
eta = 0.05		 # space discretization
tau = 0.15		 # time discretization

epsilon = 0.1    # desired precision

unsafe_radius = 0.15 # radius of unsafe set around origin

# Set random seed
np.random.seed(1)

############################################################
############################################################
############################################################

# Verify that abstraction is 0.1-approximate bisimulation
# with respect to both KL functions
assert(verify_bisim( kl1, tau, eta, epsilon))
assert(verify_bisim( kl2, tau, eta, epsilon))

# Initiate abstraction
ab = Abstraction(lb, ub, eta, tau)
# add modes to abstraction
ab.add_mode(vf1)
ab.add_mode(vf2)

# extract abstraction graph
print "abstraction has ", len(ab), " states"

# order fcn def
order_fcn = ab.node_to_idx
forbidden_nodes = [ node for node, attr in ab.graph.nodes_iter(data=True)  \
			if np.linalg.norm(attr['mid']) < unsafe_radius + epsilon ]
print len(forbidden_nodes)

# randomize an initial condition
init = np.zeros(len(ab))
j = 0
while j < 10000:
	i = np.random.randint( len(ab), size=1)
	if ab.idx_to_node(i) not in forbidden_nodes:
		init[i] += 1
		j += 1

# some random cycles
cycle_set = []
c_quot_set = set([])
for i in range(500):
	c = random_cycle(ab.graph, forbidden_nodes, 5, 0.8)
	c_quot = cyclequot(ab.graph,c,1)
	if c_quot not in c_quot_set:
		cycle_set.append(c)
		c_quot_set.add(c_quot)
print len(cycle_set)

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

pickle.dump((ab.graph, sol_data), open('example_nonlinear.save', 'wb') )
