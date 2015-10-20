"""
2D TCL example
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from numpy.linalg import norm, eig
from scipy.linalg import expm

import cPickle as pickle
import os

from modecount import *

filename = "example_tcl_2d_abstraction"

# TCL parameters
T_set = 25.0 # temperature set-point [deg C]		U([24, 26] )
T_db = 2.0 	 # temperature dead-band [deg C]		U([2, 2.5] )

Ca = 0.18  # air heat capacitance 	[kWh / deg C]  	U( [0.16, 0.21] )
Cm = 5.	   # mass heat capacitance	[kWh / deg C]	U( [4.48, 6.07] )

Hm = 0.15  # air-mass conductance	[kW / deg C]   	U( [0.2, 0.27] )
Ua = 1.0   # envelope conductance	[kW / deg C]	U( [0.84, 1.14] )
To = 32	   # outdoor temp			deg C		    32
Qa = 0.5   # air heat gain			[kW]			N( 0.5, 2.5e-9 )
Qm = 0.5   # mass heat gain 		[kW]			0.5
Qh = -15.0 # TCL heat transfer		[kW]			[-17.7, -13.1]

vf_on = lambda T: [ ( T[1] * Hm - T[0] * (Ua + Hm) + Qa + Qh + To * Ua ) / Ca,
					 ( Hm * ( T[0] - T[1] ) + Qm ) / Cm]


vf_off = lambda T: [ ( T[1] * Hm - T[0] * (Ua + Hm) + Qa + To * Ua ) / Ca,
					 ( Hm * ( T[0] - T[1] ) + Qm ) / Cm]

# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
kl_on = kl_off = lambda r,s : r * norm( expm(s*np.array([[-(Ua + Hm),  Hm], [Hm, -Hm]])) , np.inf)

# Abstraction parameters
lb = [T_set - T_db/2, T_set - 2*T_db] # lower bounds
ub = [T_set + T_db/2, T_set + 2*T_db] # upper bounds

eta = 0.001			# space discretization
tau = 0.05				# time discretization
eps = 3					# desired bisimulation approximation

# Verify that abstraction is eps-approximate bisimulation
# with respect to both KL functions
assert(verify_bisim(kl_on, tau, eta, eps))
assert(verify_bisim(kl_off, tau, eta, eps))

print "abstraction will have ", np.product((np.array(ub)-np.array(lb))/eta), " states"

# Initiate abstraction
if os.path.isfile(filename + str(eta) + str(tau) + ".save"):
	print "loading abstraction"
	ab = pickle.load( open(filename + str(eta) + str(tau) + ".save",'rb')  )
else:
	print "computing abstraction"
	ab = Abstraction(lb, ub, eta, tau)

	print "adding modes.."
	ab.add_mode(vf_on)
	ab.add_mode(vf_off)

	pickle.dump(ab, open(filename + str(eta) + str(tau) + ".save",'wb') )

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

# mode-counting synthesis
mc_sol = synthesize2(G, init, T, mode_des, mode, order_fcn = order_fcn, integer=False, verbosity = 1)

# simulate it on the connected subset of the graph!
strongly_conn_nodes = G.subgraph(max(nx.strongly_connected_components(G), key=len))
anim = simulate(G, mc_sol, strongly_conn_nodes)
anim.save('example_abstraction_lin2d_anim.mp4', fps=10)

plt.show()
