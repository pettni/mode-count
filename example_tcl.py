"""
TCL example
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import os
import cPickle as pickle

from modecount import *

filename = "example_tcl_abstraction"

# TCL parameters
Cth = 2.
Rth = 2.
Pm = 5.6
eta_tcl = 2.5
theta_r = 22.5
delta = 1

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
eta = 0.00195				# space discretization
tau = 0.05					# time discretization
eps = 0.1					# desired bisimulation approximation

# Verify that abstraction is eps-approximate bisimulation
# with respect to both KL functions
assert(verify_bisim(kl1, tau, eta, eps))
assert(verify_bisim(kl2, tau, eta, eps))


# Initiate abstraction
if os.path.isfile(filename + str(eta) + str(tau) + ".save"):
	print "loading abstraction"
	ab = pickle.load( open(filename + str(eta) + str(tau) + ".save",'rb')  )
else:
	print "computing abstraction"
	print "abstraction will have ", np.product((np.array(ub)-np.array(lb))/eta), " states"
	ab = Abstraction(lb, ub, eta, tau)

	print "adding modes"
	ab.add_mode(vf1)
	ab.add_mode(vf2)

	pickle.dump(ab, open(filename + str(eta) + str(tau) + ".save",'wb') )

# extract abstraction graph
G = ab.graph
# draw_modes(G)
# plt.show()

# bijection nodes <--> integers
order_fcn = ab.node_to_idx

strongly_conn_nodes = max(nx.strongly_connected_components(G), key=len)

# randomize an initial condition
np.random.seed(0)
init = np.zeros(len(G))
np.random.seed(1)
for k in range(10000):
	node = random.choice(list(strongly_conn_nodes))
	init[ order_fcn(node) ] += 1

# mode counting synthesis parameters
T = 5 			# horizon
mode_des = 3000	# desired mode count over time
mode = 1		# mode to count (1 or 2)


# mode-counting synthesis
mc_sol = synthesize2(G, init, T, mode_des, mode, order_fcn = order_fcn, integer = True, verbosity = 1)

# simulate it on the connected subset of the graph!
anim = simulate(G, mc_sol, order_fcn, strongly_conn_nodes)
anim.save('example_abstraction_lin2d_anim.mp4', fps=10)

plt.show()
