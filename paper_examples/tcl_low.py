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

filename = "tcl_low"

# population size
pop_size = 10000

# desired temperature interval
theta_r = 22.5
delta = 1

# mode counting synthesis parameters
T = 15 			# horizon
mode_des = 3100	# desired mode count over time
mode = 1		# mode to count (1 or 2)

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

filename_abs = filename + "_abstraction_" + str(eta) + str(tau) + ".save"
filename_data = filename + "_data_" + str(T) + "_" + str(mode) + "_" + str(mode_des) + "_" + str(pop_size) + ".save"
filename_plotdata = filename + "_plotdata_" + str(T) + "_" + str(mode) + "_" + str(mode_des) + "_" + str(pop_size) + ".save"
filename_hist = filename + "_hist_" + str(T) + "_" + str(mode) + "_" + str(mode_des) + "_" + str(pop_size) + ".save"

vf1, vf2, kl1, kl2 = tcl_model()

# Verify that abstraction is eps-approximate bisimulation
# with respect to both KL functions
assert(verify_bisim(kl1, tau, eta, eps))
assert(verify_bisim(kl2, tau, eta, eps))

############################################################
############################################################
############################################################
d_on, d_off, _, _ = tcl_model()
on_dyn = lambda z,t: d_on(z)
off_dyn = lambda z,t: d_off(z)

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
	print "loading abstraction"
	ab = pickle.load( open(filename_abs,'rb')  )
else:
	print "computing abstraction"
	print "abstraction will have ", np.product((np.array(ub)-np.array(lb))/eta), " states"
	ab = Abstraction(lb, ub, eta, tau)
	ab.add_mode(vf1)
	ab.add_mode(vf2)

	pickle.dump(ab, open(filename_abs,'wb') )

G = ab.graph
order_fcn = ab.node_to_idx

print "Graph diameter: ", nx.diameter(G)

############################################################################
########## Load / generate TCL population and control strategy #############
############################################################################

if os.path.isfile(filename_data):
	print "loading saved data"
	population, mc_sol_int = pickle.load(open(filename_data, 'rb'))
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
	mc_sol_nonint = synthesize(G, init, T, mode_des, mode, cycle_set = cycle_set, \
						 order_fcn = order_fcn, integer = False, verbosity = 1, factor=1.3)

	# make cycle set integer
	cycle_set = mc_sol_nonint['cycles']
	int_assignment = make_integer(mc_sol_nonint['assignments'])


	mc_sol_int = reach_cycles(G, init, T, mode, cycle_set = cycle_set, assignments = int_assignment, \
					     order_fcn = order_fcn, integer = True, verbosity = 3)

	pickle.dump((population, mc_sol_int), open(filename_data, 'wb') )

print "solution has transient guarantee [", min([ sum(mc_sol_int['states'][:len(G), t]) for t in range(T)]), max([ sum(mc_sol_int['states'][:len(G), t]) for t in range(T)]), "]"

print "solution has steady state guarantee", cycles_maxmin(G, mc_sol_int['cycles'], mode, mc_sol_int['assignments'])

############################################################################
##################### Perform the simulation ###############################
############################################################################

if os.path.isfile(filename_plotdata):
	print "loading saved simulation"
	xvec_disc, uvec_disc, xvec_cont, modecount_cont = pickle.load( open(filename_plotdata, 'rb') )
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

		u = controls.get_u(disc_t, xvec_disc[:,disc_t])

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

	pickle.dump((xvec_disc, uvec_disc, xvec_cont, modecount_cont), open(filename_plotdata, 'wb') )


max_t = 23.6
min_t = 21.4
num_bin_x = 200
tau = 0.05 
step = (23.6-21.4)/num_bin_x
bins = np.arange(min_t, max_t, step)
tvec = np.arange(0,30,tau)

if os.path.isfile(filename_hist):
	print "loading histogram"
	hist_vec = pickle.load(open(filename_hist, 'rb'))
else:
	hist_vec = []

	for t in range(0, len(xvec_cont)):
		print t
		hist = np.zeros(len(bins))
		for state_i in xvec_cont[t]:
			bin_index = next(i for i in range(len(bins)) if bins[i] + step/2 >= state_i[0] )
			hist[bin_index] += 1
		hist_vec.append(hist)

	hist_vec = np.array(hist_vec)

	pickle.dump(hist_vec, open(filename_hist, 'wb'))

columnwidth = 240.0/72.27

width = columnwidth

fig = plt.figure(figsize = (width,width/2))

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 10,
          'font.family' : 'lmodern',
          'text.latex.unicode': True
}

plt.pcolormesh(tvec[range(0, len(tvec)/3-1)], bins, hist_vec[range(0, len(tvec)/3-1),:].transpose(), cmap = plt.cm.Blues)
plt.ylim([min_t, max_t])
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.rcParams.update(params) 
plt.tight_layout()
plt.savefig('tcl_low_density.pdf')
