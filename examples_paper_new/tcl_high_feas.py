"""
TCL example
"""

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import cPickle as pickle

import os
import sys

sys.path.append('../')

from modecount import *
from models import tcl_model

filename = "tcl_high_feas"

# population size
pop_size = 10000

# desired temperature interval
theta_r = 22.5
delta = 1

# mode counting synthesis parameters
T = 20 			# horizon
mode = 1		# mode to count (1 or 2)
mc_lb = 3600
mc_ub = 3600	# desired mode count over time
mc_lb_prefix = 3590
mc_ub_prefix = 4600		# desired mode count over time

# Abstraction parameters
lb = [theta_r - delta]		# lower bounds
ub = [theta_r + delta]		# upper bounds
eta = 0.00195				# space discretization
tau = 0.05					# time discretization
eps = 0.1					# desired bisimulation approximation

# random seed
np.random.seed(0)

############################################################
############################################################
############################################################

# Define some filenames to save intermediate computational results

name_suffix = str(T) + "_" + str(mode) + "_" + str([mc_lb, mc_ub, mc_lb_prefix, mc_ub_prefix]) + "_" + str(pop_size) + ".save"

# save abstraction
filename_abs = filename + "_abstraction_" + str(eta) + str(tau) + ".save"

# save controller data
filename_controller = filename + "_controller_" + name_suffix

# save simulation data
filename_simulation = filename + "_simulation_" + name_suffix

filename_hist = filename + "_hist_" + name_suffix

############################################################
############################################################
############################################################

d_on, d_off, beta_on, beta_off = tcl_model()
on_dyn = lambda z,t: d_on(z)
off_dyn = lambda z,t: d_off(z)

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
	ab.add_mode(d_on)
	ab.add_mode(d_off)

	pickle.dump(ab, open(filename_abs,'wb') ) # save it!

G = ab.graph
order_fcn = ab.node_to_idx
print "Graph diameter: ", nx.diameter(G)

############################################################################
########## Load / generate TCL population and control strategy #############
############################################################################


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

mc_sol_nonint = prefix_suffix_feasible( data_nonint )

print "nonint solution has prefix mc bounds ", prefix_maxmin(G, mc_sol_nonint['states'], mode)
print "nonint solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_nonint['cycles'], mode, mc_sol_nonint['assignments'])

# make cycle set integer
int_assignment = make_avg_integer(mc_sol_nonint['assignments'])

print "rounded solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_nonint['cycles'], mode, int_assignment)

data_int = data_nonint.copy()
data_int['cycle_set'] = mc_sol_nonint['cycles']
data_int['assignments'] = int_assignment
data_int['lb_prefix'] = mc_lb_prefix
data_int['ub_prefix'] = mc_ub_prefix
data_int['ilp'] = True

mc_sol_int = prefix_feasible( data_int )

print "int solution has prefix mc bounds ", prefix_maxmin(G, mc_sol_int['states'], mode)
print "int solution has suffix mc bounds ", suffix_maxmin(G, mc_sol_int['cycles'], mode, mc_sol_int['assignments'])

pickle.dump((population, mc_sol_int), open(filename_controller, 'wb') )
print "saved solution to " + filename_controller