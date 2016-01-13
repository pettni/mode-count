"""
Example 5.2 from 

Petter Nilsson and Necmiye Ozay, 
Control synthesis for large collections of systems with mode-counting constraints,
Proceedings of the International Conference on Hybrid Systems: Computation and Control, 2016
"""

import numpy as np
import networkx as nx

import cPickle as pickle

import os
import sys

sys.path.append('../')
from modecount import *

# TCL parameters
Cth = 2.
Rth = 2.
Pm = 5.6
eta_tcl = 2.5

# Ambient temperature
theta_a = 32.

# Derived constants
a = 1./(Rth * Cth)
b = eta_tcl / Cth

# Define a vector fields
vf_on = lambda theta : -a * ( theta - theta_a ) - b * Pm  # tcl on
vf_off = lambda theta : -a * ( theta - theta_a ) 			# tcl off

# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
beta_on = lambda r,s : r * np.exp(-s*a)
beta_off = lambda r,s : r * np.exp(-s*a)

target_mc = 'low'    # 'high' or 'low'

################################################
################################################
################################################

assert target_mc in ['high', 'low']

if  target_mc == 'high':
	filename = "tcl_high_feas"
elif target_mc == 'low':
	filename = "tcl_low_feas"

# population size
pop_size = 10000

# desired temperature interval
theta_r = 22.5
delta = 1

# mode counting synthesis parameters
T = 20 			# horizon
mode = 1		# mode to count (1 or 2)

if target_mc == 'high':
	mc_lb = 3600
	mc_ub = 3600			# desired mode count over time
	mc_lb_prefix = 2500
	mc_ub_prefix = 4600		# desired mode count over time
elif target_mc == 'low':
	mc_lb = 3200
	mc_ub = 3200			# desired mode count over time
	mc_lb_prefix = 2500
	mc_ub_prefix = 4600		# desired mode count over time

# Abstraction parameters
lb = [theta_r - delta]		# lower bounds
ub = [theta_r + delta]		# upper bounds
eta = 0.00195				# space discretization
tau = 0.05					# time discretization
eps = 0.1					# desired bisimulation approximation

# set random seed for repeatability
np.random.seed(0)

############################################################
## Define some filenames to save intermediate steps ########
############################################################

name_suffix = str(T) + "_" + str(mode) + "_" + str([mc_lb, mc_ub, mc_lb_prefix, mc_ub_prefix]) + "_" + str(pop_size) + ".save"

# save abstraction
filename_abs = filename + "_abstraction_" + str(eta) + str(tau) + ".save"

# save controller data
filename_controller = filename + "_controller_" + name_suffix

# save simulation data
filename_simulation = filename + "_simulation_" + name_suffix

############################################################
############  Create a bunch of TCL's  #####################
############################################################

on_dyn = lambda z,t: vf_on(z)
off_dyn = lambda z,t: vf_off(z)

# Verify that abstraction is eps-approximate bisimulation
# with respect to both KL functions
assert(verify_bisim(beta_on, tau, eta, eps))
assert(verify_bisim(beta_off, tau, eta, eps))

class TCL(object):

	def __init__(self, init):
		self.state = init
		self.disc_state = -1 # not actual discrete state, but bisimilar counterpart
		self.mode = 'on'
	def step(self, dt):
		if self.mode == 'on':
			x_fin = scipy.integrate.odeint(on_dyn, self.state, np.arange(0, dt, dt/100))
		else:
			x_fin = scipy.integrate.odeint(off_dyn, self.state, np.arange(0, dt, dt/100))
		self.state = x_fin[-1]

############################################################
###########   Load or compute abstraction ##################
############################################################

if os.path.isfile(filename_abs):
	print "loading saved abstraction"
	ab = pickle.load( open(filename_abs,'rb')  )
else:
	print "computing abstraction"
	print "abstraction will have ", np.product((np.array(ub)-np.array(lb))/eta), " states"
	ab = Abstraction(lb, ub, eta, tau)
	ab.add_mode(vf_on)
	ab.add_mode(vf_off)

	pickle.dump(ab, open(filename_abs,'wb') ) # save it!

G = ab.graph
order_fcn = ab.node_to_idx
print "Graph diameter: ", nx.diameter(G)

############################################################################
########## Load / generate TCL population and control strategy #############
############################################################################

if os.path.isfile(filename_controller):
	print "loading saved population and controller"
	population, mc_sol_int = pickle.load( open(filename_controller,'rb')  )
	print "int solution has prefix mc bounds ", prefix_maxmin(G, mc_sol_int['states'], mode)
	print "int solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_int['cycles'], mode, mc_sol_int['assignments'])        
else:

	population = [TCL(22 + 1 * np.random.rand(1)) for i in range(pop_size)]

	init = np.zeros(len(G))
	for tcl in population:
		init[ order_fcn(ab.point_to_midx(tcl.state) ) ] += 1

	# generate a set of random cycles
	cycle_set = []
	c_quot_set = set([])
	while len(cycle_set) < 100:
		c = random_cycle(G, 5, 0.8)
		c_quot = cyclequot(G,c,mode)
		if c_quot not in c_quot_set:
			cycle_set.append(c)
			c_quot_set.add(c_quot)

	print "Min cycle ratio: ", min(c_quot_set)
	print "Max cycle ratio: ", max(c_quot_set)

	# non-integer mode-counting synthesis
	data_nonint = {}
	data_nonint['graph'] = G
	data_nonint['init'] = init
	data_nonint['horizon'] = T

	data_nonint['mode'] = mode
	data_nonint['lb_suffix'] = mc_lb
	data_nonint['ub_suffix'] = mc_ub
	data_nonint['lb_prefix'] = mc_lb_prefix
	data_nonint['ub_prefix'] = mc_ub_prefix

	data_nonint['cycle_set'] = cycle_set

	data_nonint['order_function'] = order_fcn
	data_nonint['ilp'] = False

	mc_sol_nonint = prefix_suffix_feasible( data_nonint, solver='gurobi' )

	print "nonint solution has prefix mc bounds ", prefix_maxmin(G, mc_sol_nonint['states'], mode)
	print "nonint solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_nonint['cycles'], mode, mc_sol_nonint['assignments'])

	# make cycle set integer
	int_assignment = make_avg_integer(mc_sol_nonint['assignments'])

	print "rounded solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_nonint['cycles'], mode, int_assignment)

	# integer mode-counting synthesis
	data_int = data_nonint.copy()
	data_int['cycle_set'] = mc_sol_nonint['cycles']
	data_int['assignments'] = int_assignment
	data_int['lb_prefix'] = mc_lb_prefix
	data_int['ub_prefix'] = mc_ub_prefix
	data_int['ilp'] = True

	mc_sol_int = prefix_feasible( data_int , solver='gurobi')

	print "int solution has prefix mc bounds ", prefix_maxmin(G, mc_sol_int['states'], mode)
	print "int solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_int['cycles'], mode, mc_sol_int['assignments'])

	pickle.dump((population, mc_sol_int), open(filename_controller, 'wb') )
	print "saved solution to " + filename_controller

#############################################
########## Perform a simulation #############
#############################################

if os.path.isfile(filename_simulation):
	print "loading saved simulation"
	xvec_disc, uvec_disc, xvec_cont, modecount_cont = pickle.load( open(filename_simulation, 'rb') )
else:
	A,B = lin_syst(G, order_fcn)
	controls = CycleControl(G, mc_sol_int, order_fcn)

	disc_init = mc_sol_int['states'][:,0]

	tmax = 10
	disc_tmax = int(tmax / tau)

	xvec_disc = np.zeros([len(disc_init), disc_tmax+1], dtype=float)
	uvec_disc = np.zeros([len(disc_init), disc_tmax+1], dtype=float)
	xvec_disc[:,0] = disc_init

	xvec_cont = []
	modecount_cont = []

	# set TCL discrete state and create lists
	# for easy access to "all TCLs in a certain node"
	vertex_to_tcls = [ [] for i in range(len(G)) ]
	for i, tcl in enumerate(population):
		tcl.disc_state = order_fcn( ab.point_to_midx(tcl.state) )
		vertex_to_tcls[tcl.disc_state].append(i)

	for disc_t in range(disc_tmax):
		print "TIME: ", disc_t, " out of ", disc_tmax

		u = controls.get_u(disc_t, xvec_disc[:,disc_t], solver='gurobi')

		flows = xvec_disc[:,disc_t] - u + scipy.sparse.bmat( [ [None, scipy.sparse.identity(len(G))], [scipy.sparse.identity(len(G)), None] ] ).dot( u )

		# assign modes
		for i, tcls in enumerate(vertex_to_tcls):
			num_mode_on = round(flows[i])
			num_mode_off = round(flows[i+len(G)])

			if num_mode_on:
				assert len( [new_node for new_node in G.successors( ab.idx_to_node(i) ) if G[ab.idx_to_node(i)][new_node]['mode'] == 1] ) >= 1 
			if num_mode_off:
				assert len( [new_node for new_node in G.successors( ab.idx_to_node(i) ) if G[ab.idx_to_node(i)][new_node]['mode'] == 2] ) >= 1

			for j, tcl_index in enumerate(tcls):
				if j < num_mode_on:
					population[tcl_index].mode = 'on'
				else:
					population[tcl_index].mode = 'off'

		# simulate continuous time
		step = 1
		for i in range(step):
			dt = tau/step
			for tcl in population: tcl.step(dt)
			xvec_cont.append(np.array([ tcl.state for tcl in population ]))
			modecount_cont.append( sum( [int(tcl.mode == 'on') for tcl in population]  ) )

		# update discrete time
		for i, tcl in enumerate(population):
			current_node = ab.idx_to_node(tcl.disc_state)
			mode_num = 1 if tcl.mode == 'on' else 2
			next_disc_state = order_fcn([next_node for next_node in G.successors(current_node) if G[current_node][next_node]['mode'] == mode_num][0])

			vertex_to_tcls[tcl.disc_state].remove(i)
			vertex_to_tcls[next_disc_state].append(i)

			tcl.disc_state = next_disc_state

		uvec_disc[:,disc_t] = u
		xvec_disc[:,disc_t+1] = A.dot(xvec_disc[:,disc_t]) + B.dot(u)

		assert( abs(sum(xvec_disc[:, disc_t+1]) - 10000) < 1e-5)

	pickle.dump((xvec_disc, uvec_disc, xvec_cont, modecount_cont), open(filename_simulation, 'wb') )
