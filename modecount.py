"""
Module for computing abstractions and performing mode-counting synthesis
on the abstraction.

Classes:
	- Abstraction, represents an abstraction of a switched dynamical system
	- CycleControl, computes mode-counting enforcing feedback controls

Main methods:
	- lin_syst: returns a linear system description (A,B) of the aggregate dynamics on a graph
	- synthesize: solve a mode-counting problem on a mode-graph
	- simulate: simulate a synthesizes solution using matplotlib.animation

Prerequisites:
	- numpy
	- scipy
	- matplotlib
	- networkx
	- mosek configured for Python (tested with version 7.1.0.40)

"""

from collections import deque

import numpy as np

import scipy.integrate
import scipy.sparse
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx
import itertools

import time

from random_cycle import random_cycle
from optimization_wrappers import solve_mip, solve_lp

np.set_printoptions(precision=4, suppress=True)

class Abstraction(object):
	""" 
		Discrete abstraction of the hyper box defined by *lower_bounds* and *upper_bounds*.
		Time discretization is *tau* and space discretization is given by *eta*.
		Mode transitions are added with :py:func:`add_mode`.
	"""
	def __init__(self, lower_bounds, upper_bounds, eta, tau):
		self.eta = eta
		self.tau = float(tau)

		self.lower_bounds = np.array(lower_bounds, dtype=np.float64)
		self.upper_bounds = np.array(upper_bounds, dtype=np.float64)
		self.upper_bounds += (self.upper_bounds - self.lower_bounds) % eta # make domain number of eta's
		self.nextMode = 1

		# number of discrete points along each dimension
		self.n_dim = np.round((self.upper_bounds - self.lower_bounds)/eta).astype(int)

		# Create a graph and populate it with nodes
		self.graph = nx.DiGraph()
		for midx in itertools.product(*[range(dim) for dim in self.n_dim]):
			cell_lower_bounds = self.lower_bounds + np.array(midx) * eta
			cell_upper_bounds = cell_lower_bounds + eta
			self.graph.add_node(midx, lower_bounds = tuple(cell_lower_bounds), upper_bounds = tuple(cell_upper_bounds), mid = (cell_lower_bounds + cell_upper_bounds) / 2)

	def __len__(self):
		''' Size of abstraction '''
		return len(self.graph)

	def point_to_midx(self, point):
		'''	Return the node multiindex corresponding to the continuous point *point*. '''
		assert(len(point) == len(self.n_dim))
		if not self.contains(point):
			raise('Point outside domain')
		midx = np.floor((np.array(point) - self.lower_bounds) / self.eta).astype(np.uint64)
		return tuple(midx)

	def contains(self, point):
		''' Return ``True`` if *point* is within the abstraction domain, ``False`` otherwise. '''
		if np.any(self.lower_bounds >= point) or np.any(self.upper_bounds <= point):
			return False
		return True

	def plot_planar(self):
		'''	Plot a 2D abstraction. '''
		assert(len(self.lower_bounds) == 2)
		ax = plt.axes()
		for node, attr in self.graph.nodes_iter(data = True):
			plt.plot(attr['mid'][0], attr['mid'][1], 'ro')
		for n1, n2, mode in self.graph.edges_iter(data='mode'):
			mid1 = self.graph.node[n1]['mid']
			mid2 = self.graph.node[n2]['mid']
			col = 'b' if mode == 1 else 'g'
			ax.arrow(mid1[0], mid1[1],  mid2[0] - mid1[0], mid2[1] - mid1[1], fc=col, ec=col)

		plt.show()

	def add_mode(self, vf):
		'''	Add new dynamic mode to the abstraction, given by the vector field *vf*. '''
		dummy_vf = lambda z,t : vf(z)
		tr_out = 0
		for node, attr in self.graph.nodes_iter(data=True):
			x_fin = scipy.integrate.odeint(dummy_vf, attr['mid'], np.arange(0, self.tau, self.tau/100))
			if self.contains(x_fin[-1]):
				midx = self.point_to_midx(x_fin[-1])
				self.graph.add_edge( node, midx, mode = self.nextMode )
			else:
				tr_out += 1
		if tr_out > 0:
			print "Warning: ", tr_out, " transitions out of ", len(self.graph), " in mode ", self.nextMode, " go out of domain"
		self.nextMode += 1

	def node_to_idx(self, node):
		''' Given a node at discrete multiindex :math:`(x,y,z)`, return the 
			index :math:`L_z ( L_y x + y ) + z`, 
			where :math:`L_z, L_y` are the (discrete) lengths of the hyper box domain,
			and correspondingly for higher/lower dimensions. The function is a 1-1 mapping between
			the nodes in the abstraction and the positive integers, and thus suitable as 
			order_function in :py:func:`prefix_suffix_feasible`. '''
		assert len(node) == len(self.n_dim)
		ret = node[0]
		for i in range(1,len(self.n_dim)):
			ret *= self.n_dim[i]
			ret += node[i]
		return ret

	def idx_to_node(self, idx):
		''' Inverse of :py:func:`node_to_idx` '''
		assert(idx < np.product(self.n_dim))
		node = [0] * len(self.n_dim)
		for i in reversed(range(len(self.n_dim))):
			node[i] = int(idx % self.n_dim[i])
			idx = np.floor(idx / self.n_dim[i])
		return tuple(node)

class CycleControl():
	def __init__(self, G, sol, order_fcn):
		self.G = G
		self.u_arr = sol['controls']
		self.c_list = sol['cycles']
		self.alpha_list = [deque(a) for a in sol['assignments']]
		self.order_fcn = order_fcn
		self.A, self.B = lin_syst(G, order_fcn)
		self.stacked_e = _stacked_eye(G)

	def get_u(self, t, state, solver=None):

		if t < self.u_arr.shape[1]:
			return self.u_arr[:,t]

		# compute desired next state
		for a in self.alpha_list: a.rotate()
		
		# solve linear system to find control
		sum_next = np.sum([_cycle_indices(self.G, c, self.order_fcn).dot( a ) for c,a in zip(self.c_list, self.alpha_list)], 0)
		beq = sum_next - self.stacked_e.dot(self.A.dot( state ) )
		Aeq = self.stacked_e.dot( self.B )
	
		Aiq = scipy.sparse.bmat( [ [-scipy.sparse.identity(self.B.shape[1])],
								   [scipy.sparse.identity(self.B.shape[1])] ] )
		biq = np.hstack([np.zeros(self.B.shape[1]), state])

		c = np.zeros(self.B.shape[1])
		sol = solve_lp(c, Aiq, biq, Aeq, beq, solver=solver)
		return np.array(sol['x']).flatten()

def verify_bisim(beta, tau, eta, eps):
	''' Given a :math:`\mathcal{KL}`-function *beta* for a continuous system, return ``True`` if the abstraction
	created with time discretization *tau* and space discretization *eta* passes the *eps*-approximate bisimilarity
	test, and ``False`` otherwise. '''
	return (beta(eps, tau) + eta/2 <= eps)

def lin_syst(G, order_fcn = None):
	''' Given a graph *G* with edges labeled with integers :math:`1, ..., M`, compute
	 	matrices :math:`A,B` such that 

	 	.. math:: \mathbf w(t+1) = A \mathbf w + B \mathbf r,

	 	where :math:`w_i` represents the number of individual systems at node :math:`n` in mode :math:`m` if

	 	.. math:: i = (m-1) K + order\_fcn(n).

	 	If no order function is specified, ordering by **G.nodes().index** is used.
	'''
	if order_fcn == None:
		order_fcn = lambda v : G.nodes().index(v)

	ordering = sorted(G.nodes_iter(), key = order_fcn)

	adj_data = nx.to_dict_of_dicts(G)
	
	T_list = []
	for mode in range(1, _maxmode(G)+1):
		data = np.array([(1, order_fcn(node2), order_fcn(node1)) \
								for (node1, node1_out) in adj_data.iteritems()  \
								for node2 in node1_out \
								if node1_out[node2]['mode'] == mode])
		T_mode = scipy.sparse.coo_matrix( (data[:,0], (data[:,1], data[:,2])), shape=(len(G), len(G)) )
		T_list.append(T_mode)

	A = scipy.sparse.block_diag(tuple(T_list), dtype=np.int8)	
	B = scipy.sparse.bmat([ [Ti for i in range(len(T_list))] for Ti in T_list ]) - 2*A

	return A,B

def prefix_feasible(problem_data, verbosity = 1, solver=None):
	'''
    Define and solve the prefix part of a mode-counting 
    synthesis problem (requires a given suffix part)
     
    Inputs:

    * *problem_data*: dictionary with the following fields:

      * ``'graph'``			: mode-transition graph G
      * ``'init'``			: initial configuration in G    
      * ``'horizon'``		: length of strategy prefix part
      * ``'cycle_set'``     : cycles to form suffix part
      * ``'assignments'``   : assignments to ``'cycle_set'``
      * ``'mode'``          : mode to count
      * ``'lb_prefix'``     : lower mode-counting bound in prefix phase
      * ``'ub_prefix'``     : upper mode-counting bound in prefix phase

      Optional fields

      * ``'order_function'`` : a function that is a bijection that maps nodes in G to integers :math:`[0, 1, \ldots , N]`  (default: G.nodes().index)
      * ``'forbidden_nodes'``: nodes in G that can not be visited (default: [])
      * ``'ilp'``            : if true, solve as ILP (default: ``True``)

    * *verbosity*: level of verbosity
        
    Output: a dictionary with the following fields:

    * ``'controls'``       : prefix part u of strategy, u[:,t] is control at time t
    * ``'states'``         : states x generated by u, x[:,t] is state at time t
    * ``'cycles'``         : suffix cycles (same as input)
    * ``'assignments'``    : suffix assignments (same as input)
	'''
	G, T, N, ilp, order_fcn, forbidden_nodes = _extract_arguments(problem_data)

	# variables: u[0], ..., u[T-1], x[0], ..., x[T]
	N_u 	= T * N 			# input vars
	N_x     = (T+1) * N 		# state vars

	N_tot = N_u + N_x

	if verbosity: lp_start = time.time(); print "Setting up LP.."
	
	################################
	# Initialize basic constraints #
	################################
	Aeq, beq, Aiq, biq = _ux_constraints(G, problem_data['init'], T, order_fcn, forbidden_nodes)

	###############################
	##### Time T constraints ######
	###############################
	Psi_mats = [_cycle_indices(G, cycle, order_fcn) for cycle in problem_data['cycle_set']]
	Aeq = _sparse_vstack( Aeq, _stacked_eye(G), N_u + N*T )
	beq = np.hstack([beq, np.sum( [ _cycle_indices(G, cycle, order_fcn).dot(ass) for cycle, ass in zip( problem_data['cycle_set'], problem_data['assignments'] ) ], 0)])

	###############################
	##### Prefix mode-count #######
	###############################
	Aiq_new, biq_new = _prefix_mc(G, T, N, problem_data['mode'], problem_data['lb_prefix'], problem_data['ub_prefix'])
	Aiq = _sparse_vstack(Aiq, Aiq_new, N_u )
	biq = np.hstack([biq, biq_new])
	
	if verbosity >= 1: print "It took ", time.time() - lp_start, " to set up (I)LP"

	##############################################################
	##############################################################
	##############################################################

	if verbosity >= 1: print "solving (I)LP..."; solve_start = time.time()
	
	if ilp: lp_sln = solve_mip(np.zeros(N_tot), Aiq, biq, Aeq, beq, set(range(N_u)), solver=solver)
	else: lp_sln = solve_lp(np.zeros(N_tot), Aiq, biq, Aeq, beq, solver=solver)
	
	if verbosity >= 1: print "It took ", time.time() - solve_start, " to solve (I)LP"

	sol = _extract_solution_state(lp_sln['x'], N, T)
	sol['assignments'] = problem_data['assignments']
	sol['cycles'] = problem_data['cycle_set']

	if verbosity >= 2:
		_print_sol(G, problem_data['mode'], sol)

	return sol

def prefix_suffix_feasible(problem_data, verbosity = 1, solver=None):
	'''
    Define and solve a mode-counting synthesis problem
    with a prefix-suffix strategy.
     
    Inputs:
        
    * *problem_data*: dictionary with the following fields:

      * ``'graph'``			: mode-transition graph G
      * ``'init'``			: initial configuration in G    
      * ``'horizon'``       : length of strategy prefix part
      * ``'cycle_set'``     : set of cycles from which suffix part is formed
      * ``'mode'``          : mode to count
      * ``'lb_suffix'``     : lower mode-counting bound in suffix phase
      * ``'ub_suffix'``     : upper mode-counting bound in suffix phase

      Optional fields

      * ``'lb_prefix'``       : lower mode-counting bound in prefix phase (default: lb_suffix)
      * ``'ub_prefix'``       : upper mode-counting bound in prefix phase (default: ub_suffix)
      * ``'order_function'``  : a function that is a bijection that maps nodes in G to integers :math:`[0, 1, \ldots , N]`  (default: G.nodes().index)
      * ``'forbidden_nodes'`` : nodes in G that can not be visited (default: [])
      * ``'ilp'``             : if true, solve as ILP (default: ``True``)

    * *verbosity*: level of verbosity

    Output: a dictionary with the following fields:

    * ``'controls'``        : prefix part u of strategy, u[:,t] is control at time t
    * ``'states'``          : states x generated by u, x[:,t] is state at time t
    * ``'cycles'``          : suffix cycles
    * ``'assignments'``     : suffix assignments
     
    Example: see example_simple.py
    '''

	G, T, N, ilp, order_fcn, forbidden_nodes = _extract_arguments(problem_data)

	# clean cycle set from forbidden nodes
	cycle_set = _clean_cycle_set(problem_data['cycle_set'], forbidden_nodes)

	# variables: u[0], ..., u[T-1], x[0], ..., x[T], a[0], ..., a[C-1], 
	#		     lb[0], ..., lb[C-1], ub[0], ... ub[C-1]
	N_u 	= T * N				# input vars
	N_x     = (T+1) *N 			# state vars
	N_cycle_tot = sum([len(cycle) for cycle in cycle_set])	# cycle vars
	N_bound 	= len(cycle_set)	# lb/ub vars

	N_tot = N_u + N_x + N_cycle_tot + 2 * N_bound

	print "Setting up LP.. \n"
	lp_start = time.time()

	################################
	# Initialize basic constraints #
	################################
	Aeq, beq, Aiq, biq = _ux_constraints(G, problem_data['init'], T, order_fcn, forbidden_nodes)
	Aeq = scipy.sparse.bmat([[ Aeq, _coo_zeros(Aeq.shape[0], N_cycle_tot + 2 * N_bound) ]])
	Aiq = scipy.sparse.bmat([[ Aiq, _coo_zeros(Aiq.shape[0], N_cycle_tot + 2 * N_bound) ]])

	###############################
	##### Time T constraints ######
	###############################
	Psi_mats = [_cycle_indices(G, cycle, order_fcn) for cycle in cycle_set]
	Aeq = _sparse_vstack( Aeq, scipy.sparse.bmat([[_stacked_eye(G), -scipy.sparse.bmat([ Psi_mats ])]]), N_u + N*T )
	beq = np.hstack([beq, np.zeros(len(G))])

	###############################
	##### Prefix mode-count #######
	###############################
	try:
		lb_prefix = problem_data['lb_prefix']
		ub_prefix = problem_data['ub_prefix']
	except:
		lb_prefix = problem_data['lb_suffix']
		ub_prefix = problem_data['ub_suffix']

	Aiq_new, biq_new = _prefix_mc(G, T, N, problem_data['mode'], lb_prefix, ub_prefix)
	Aiq = _sparse_vstack(Aiq, Aiq_new, N_u )
	biq = np.hstack([biq, biq_new])

	###############################
	##### Suffix mode-count #######
	###############################

	# Individual cycles bounds
	Aiq_new, biq_new = _suffix_mc(G, cycle_set, problem_data['mode'])
	Aiq = _sparse_vstack(Aiq, Aiq_new, N_u + N_x)
	biq = np.hstack([biq, biq_new])

	# Aggregate cycle bounds: sum(ub) < ub_tot, lb_tot < sum(lb)
	Aiq_new = scipy.sparse.block_diag( (-np.ones([1, N_bound]), np.ones([1, N_bound])) )
	Aiq = _sparse_vstack(Aiq, Aiq_new, N_u + N_x + N_cycle_tot)
	biq = np.hstack([biq, np.array([ -problem_data['lb_suffix'], problem_data['ub_suffix'] ]) ])

	###############################
	#### Positive assignments #####
	###############################
	Aiq = _sparse_vstack(Aiq, -scipy.sparse.identity( N_cycle_tot  ), N_u + N_x)
	biq = np.hstack([biq, np.zeros(N_cycle_tot )])

	if verbosity >= 1: time.time(); print "It took ", time.time() - lp_start, " to set up (I)LP"

	##############################################################
	##############################################################
	##############################################################

	if verbosity >= 1: print "solving (I)LP..."; solve_start = time.time()
	
	if ilp:
		lp_sln = solve_mip(np.zeros(N_tot), Aiq, biq, Aeq, beq, set(range(N_u)), solver=solver)
	else:
		lp_sln = solve_lp(np.zeros(N_tot), Aiq, biq, Aeq, beq, solver=solver)
	
	if verbosity >= 1: print "It took ", time.time() - solve_start, " to solve (I)LP"

	sol = _extract_solution(lp_sln['x'], N, T, cycle_set)

	if verbosity >= 2:
		_print_sol(G, problem_data['mode'], sol)

	return sol

def simulate(G, sol, order_fcn, nodelist = []):

	if len(nodelist) == 0:
		nodelist = G.nodes()

	# Construct subgraph we want to plot
	subgraph = G.subgraph(nodelist)
	subgraph_indices = [order_fcn(node) for node in subgraph.nodes()]

	# Initiate plot
	fig = plt.figure()
	ax = fig.gca()

	# Get linear system description
	A,B = lin_syst(G, order_fcn)
	maxmode = _maxmode(G)

	# feedback
	controls = CycleControl(G, sol, order_fcn)

	# Pre-compute plotting data
	tmax = 30
	ANIM_STEP = 10

	init = sol['states'][:,0]
	xvec = np.zeros([len(init), tmax+1], dtype=float)
	uvec = np.zeros([len(init), tmax+1], dtype=float)
	xvec[:,0] = init
	for t in range(tmax):
		u = controls.get_u(t, xvec[:,t])
		uvec[:,t] = u
		xvec[:,t+1] = A.dot(xvec[:,t]) + B.dot(u)

	#################################
	######### PLOT THE GRAPH ########
	#################################
	pos = nx.spring_layout(subgraph)

	time_template = 'time = %d'
	time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
	mode_template = 'mode %d count = %.1f'

	# Plot edges
	edgelists = [ [(u,v) for (u,v,d) in subgraph.edges(data=True) if d['mode'] == mode ]  for mode in range(1, maxmode+1)]
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(edgelists))))
	mode_text = []
	for i, edgelist in enumerate(edgelists):
		col = next(colors)
		nx.draw_networkx_edges(subgraph, pos, ax=ax, edgelist = edgelist, edge_color = [col] * len(edgelist))
		mode_text.append(ax.text(0.35 + i*0.35, 0.9, '', transform=ax.transAxes, color = col))

	# Plot initial set of nodes
	node_size = 300 * np.ones(len(subgraph_indices))
	
	nod = nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size = node_size )

	edge_anim = [ (v, w, plt.plot(pos[v], pos[w], marker='o', color = 'w', markersize=10) ) for v,w in subgraph.edges_iter()]

	def _sum_modes(state, maxmode):
		return np.sum([state[ i * len(state)/maxmode : (i+1) * len(state)/maxmode] for i in range(maxmode)], axis=0)

	# Function that updates node colors
	def update(i):
		x_ind = i/ANIM_STEP
		anim_ind = i % ANIM_STEP

		x_i = xvec[:,x_ind]
		u_i = uvec[:,x_ind]
		t = float(anim_ind)/ANIM_STEP

		norm = mpl.colors.Normalize(vmin=0, vmax=np.max(x_i))
		cmap = plt.cm.Blues
		m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		nod.set_facecolors(m.to_rgba(_sum_modes(x_i, maxmode)[subgraph_indices] ))

		time_text.set_text(time_template % x_ind)
		for j, mt in enumerate(mode_text):
			mt.set_text(mode_template % (j+1 , sum(xvec[range(len(G)*j, len(G)*(j+1) ), x_ind])))

		for v, w, edge_marker in edge_anim:
			ind = _stateind(G, v, G[v][w]['mode'], order_fcn)
			x_this = x_i[ ind ] - u_i[ ind ] + u_i[(ind + len(G)) % (2*len(G)) ]
			if x_this > 1e-5:
				edge_marker[0].set_visible( True )
				edge_marker[0].set_color( m.to_rgba( x_this ) )
				edge_marker[0].set_data( t * pos[w][0] + (1-t) * pos[v][0], t * pos[w][1] + (1-t) * pos[v][1] )
			else:
				edge_marker[0].set_visible( False )

		return nod, time_text, mode_text, edge_anim

	ani = animation.FuncAnimation(fig, update, tmax * ANIM_STEP, interval=10, blit=False)
	return ani

################################################################
###### Functions used in treating input data #######
################################################################

def _clean_cycle_set(cycle_set, forbidden_nodes):
	'''
	Purge cycles in 'cycle_set' that contain 'forbidden_nodes'
	'''
	new_cycle_set = []

	forbidden_nodes_set = set(forbidden_nodes)
	for cycle in cycle_set:
		if len( forbidden_nodes_set.intersection(set(cycle))) == 0:
			new_cycle_set.append(cycle) 

	return new_cycle_set

def _extract_arguments(problem_data):
	'''
	Helper function to treat problem data input.
	'''
	G = problem_data['graph']
	T = problem_data['horizon']
	N = len(G) * _maxmode(G)

	# Integer program?
	try: ilp = problem_data['ilp']
	except: ilp = True

	try: order_fcn = problem_data['order_function']
	except: print "no order function provided, may be slow"; order_fcn = G.nodes().index

	try: forbidden_nodes = problem_data['forbidden_nodes'] 
	except: print "no forbidden nodes given"; forbidden_nodes = []

	return G, T, N, ilp, order_fcn, forbidden_nodes

################################################################
###### Functions used to construct linear (in)equalities #######
################################################################

def _ux_constraints(G, init, T, order_fcn, forbidden_nodes = []):
	'''	 
	Computes Aeq, beq, Aiq, biq, s.t.
		Aeq X = beq, Aiq X <= biq
	enforces the following constraints for control and state
	  - initial state
	  - no exit
	  - dynamics
	  - control bounds
	  - forbidden nodes,
	for the variable vector	  
		X = [ u[0], ..., u[T-1], x[0], ..., x[T] ]
	'''
	A,B = lin_syst(G, order_fcn)
	maxmode = _maxmode(G)
	N = A.shape[1]
	assert(len(G) == N / maxmode)

	N_u 	= T * N 			# input vars
	N_x     = (T+1) * N 		# state vars

	###############################
	#### Dynamics constraints #####
	###############################
	N_eq_dyn = N * T 	  # dynamics
	Aeq = _dyn_eq_mat(A,B,T)
	beq = np.zeros( N_eq_dyn )

	###############################
	##### Initial constraints #####
	###############################
	Aeq = _sparse_vstack( Aeq, _stacked_eye(G), N_u )
	beq = np.hstack( [beq, init ] )

	###############################
	########### No exit ###########
	###############################
	Aeq = _sparse_vstack(Aeq, np.ones([1,N]), N_u + N*T)
	beq = np.hstack([beq, [np.sum(init)]])

	###############################
	#### No forbidden nodes #######
	###############################
	if len(forbidden_nodes) > 0:
		N_eq_forbidden = (T+1) * maxmode * len(forbidden_nodes) # forbidden nodes constraint
		Aeq = _sparse_vstack(Aeq, 
			scipy.sparse.coo_matrix( (np.ones(N_eq_forbidden), 
									 	(range(N_eq_forbidden) , 
			 							[ t * N + _stateind(G, f_node, m, order_fcn) for f_node in forbidden_nodes for t in range(T+1) for m in range(1, maxmode+1)  ]  
										) 
									 ),	
			 						(N_eq_forbidden, N_x) ), 
						    N_u )
		beq = np.hstack([beq, np.zeros(N_eq_forbidden)])

	###############################
	#### Lower control bounds #####
	###############################
	Aiq = scipy.sparse.bmat([[-scipy.sparse.identity( N_u ), _coo_zeros(N_u, N_x)]])
	biq = np.zeros(N_u)

	###############################
	#### Upper control bounds #####
	###############################
	Aiq = _sparse_vstack(Aiq, 
			scipy.sparse.bmat([[scipy.sparse.identity(N_u), 
							   -scipy.sparse.identity(N_u)]]),
			0)
	biq = np.hstack([biq, np.zeros(N_u)])

	return Aeq, beq, Aiq, biq

def _prefix_mc(G, T, N, mode, lb, ub):
	'''Compute Aiq, biq s.t. if
		Aiq X <= biq,
	mode-counts during prefix phase are enforced, for variable vector
		X = [ x[0], ..., x[T] ]
	'''

	N_ineq_mc_prefix = T+1
	sum_mode_mat = scipy.sparse.coo_matrix( ( np.ones(len(G)), ( np.zeros(len(G)), [(mode - 1)*len(G) + i for i in range(len(G))] ) ), (1, N) )
	prefix_mc_x_block = scipy.sparse.block_diag( (sum_mode_mat,) * (T+1) )

	return scipy.sparse.bmat([[-prefix_mc_x_block], [prefix_mc_x_block]]), \
		   np.hstack([ -lb * np.ones(N_ineq_mc_prefix), ub * np.ones(N_ineq_mc_prefix) ])

def _suffix_mc(G, cycle_set, mode):
	'''Compute Aiq, biq s.t. if
		Aiq X <= biq,
	cycle-wise mode-counts for cycle i during suffix phase are bounded in 
	[ lb[i], ub[i] ], for variable vector
		X = [a[0], ..., a[C-1], lb[0], ..., lb[C-1], ub[0], ..., ub[C-1]]
	'''

	# number of inequalities for lb/ ub
	N_ineq_cycle_bound = sum([len(cycle) for cycle in cycle_set])
	N_bounds = len(cycle_set)

	cycle_matrix = scipy.sparse.block_diag( tuple( [ _cycle_matrix( G,c,mode ) for c in cycle_set ] ) )
	bound_matrix = scipy.sparse.block_diag( tuple( [ np.ones([len(c), 1]) for c in cycle_set ] ) )
	zero_matrix = _coo_zeros( N_ineq_cycle_bound, N_bounds )

	# lower cycle bounds: lb_i < cycle_matrix_i * ass_i
	# upper cycle bounds: cycle_matrix_i * ass_i < ub_i

	Aiq = scipy.sparse.bmat([[-cycle_matrix, bound_matrix, zero_matrix], 
					         [cycle_matrix, zero_matrix, -bound_matrix]])
	biq = np.zeros(2 * N_ineq_cycle_bound)

	return Aiq, biq

def _dyn_eq_mat(A, B, T):
	''' 
	compute matrix Aeq s.t. if
		Aeq X = 0 
	for variable vector
		X = [ u[0], ..., u[T-1], x[0], ..., x[T] ],
	then
	 	x[t+1] = A x[t] + B u[t] 
	for t = 0, ..., T-1.
	'''
	N = A.shape[1]
	m = B.shape[1]

	N_eq_dyn = N * T

	A_dyn_u = scipy.sparse.block_diag((B,) * T)
	A_dyn_x = scipy.sparse.bmat( [[scipy.sparse.block_diag((A,) * T), scipy.sparse.coo_matrix( (N_eq_dyn, N) ) ]]) \
		+  scipy.sparse.bmat( [[scipy.sparse.coo_matrix( (N_eq_dyn, N) ), -scipy.sparse.identity( N_eq_dyn ) ]]) \

	return scipy.sparse.bmat([[A_dyn_u, A_dyn_x]])

def _sparse_vstack(sparse_mat, block, col_pos):
	'''
	Constructs the matrix
	[  sparse_mat; 
	   0   block   0 ],
	s.t. block starts at column 'col_pos' 
	'''
	ncol = sparse_mat.shape[1]
	nrow = block.shape[0]

	if col_pos > 0:
		newblock = scipy.sparse.bmat([[ _coo_zeros(nrow, col_pos), block ]])
	else:
		newblock = block

	if ncol - col_pos - block.shape[1] > 0:
		newblock = scipy.sparse.bmat([[ newblock, _coo_zeros(nrow, ncol - col_pos - block.shape[1]) ]])

	return scipy.sparse.bmat([ [ sparse_mat ], 
							   [ newblock   ] ])

def _stacked_eye(G):
	'''
	Constructs the N x (MN) matrix of identity matrices [ I I ... I ], 
	where N is the number of nodes in G and M the number of nodes
	'''
	n_v = len(G)
	n = len(G) * _maxmode(G)
	return scipy.sparse.coo_matrix( (np.ones(n), ([ (i % n_v ) for i in range(n)], range(n) )), shape = (n_v, n) )

def _coo_zeros(nrow,ncol):
	'''
	Create a scipy.sparse zero matrix with 'nrow' rows and 'ncol' columns
	'''
	return scipy.sparse.coo_matrix( (nrow,ncol) )

################################################################
###### Functions used to handle solutions #######
################################################################

def cyclequot(G, cycle, mode):
	'''
	For a given 'cycle', compute the proportion of nodes that are in 'mode' 
	'''
	return float(sum(next(_cycle_rows(G, cycle, mode))))/len(cycle)

def cycle_maxmin(G, cycle, mode, assignment):
	'''
	For a given 'cycle' and 'assignment', compute mode-counting bounds for 'mode'
	'''
	prod = np.dot(np.array(_cycle_matrix(G,cycle,mode)), assignment)
	return np.min(prod), np.max(prod)

def suffix_maxmin(G, cycles, mode, assignments):
	'''
	Given a suffix strategy consisting of `cycles` and `assignments`, compute upper and lower mode-counting bounds
	for the mode `mode`.
	'''
	return sum(np.array([cycle_maxmin(G, c, mode, a) for c,a in zip(cycles, assignments)]), 0)

def prefix_maxmin(G, states, mode):
	'''
	Given a sequence of 'states', compute upper and lower mode-counting bounds
	for the mode `mode`.
	'''
	modecount = np.sum(states[(mode-1)*len(G):mode*len(G), :], 0)
	return [np.min(modecount), np.max(modecount)]

def _print_sol(G, mode, sol):

	for cycle, ass in zip(sol['cycles'], sol['assignments']):
		print ""
		print "Cycle: ", cycle
		print "Cycle mode-count: ", next(_cycle_rows(G,cycle, mode))
		print "Assignment: ", np.array(ass)
		print "Sum: ", sum(ass)
		print "Bounds: ", np.array(cycle_maxmin(G, cycle, mode, ass))

	print ""
	print "Prefix interval: ", np.array(prefix_maxmin(G, sol['states'], mode))
	print "Suffix interval: ", np.array(suffix_maxmin(G, sol['cycles'], mode, sol['assignments']))
	print ""

def _extract_solution_state(sol_vector, N, T):
	N_u 	= T * N 			# input vars
	N_x     = (T+1) * N 		# state vars

	sol = {}
	sol['controls'] = np.array(sol_vector[:N_u]).reshape(T,N).transpose()
	sol['states'] = np.array(sol_vector[N_u:N_u+N_x]).reshape(T+1,N).transpose()
	return sol

def _extract_solution(sol_vector, N, T, cycle_set):
	N_u 	= T * N 			# input vars
	N_x     = (T+1) * N 		# state vars

	sol = _extract_solution_state(sol_vector, N, T)
	sol['cycles'] = []
	sol['assignments'] = []

	index_ai =  lambda i : N_u + N_x + sum([len(cycle_set[ii]) for ii in range(i) ])  	# starting index for a[i], i = 0, ..., C-1
	for i in range(len(cycle_set)):
		alpha_i = list(sol_vector[index_ai(i): index_ai(i) + len(cycle_set[i])])
		if sum( alpha_i ) > 1e-5:
			sol['cycles'].append(cycle_set[i])
			sol['assignments'].append(alpha_i)
	return sol

################################################################
##### Functions pertaining to graph/linear system indexing #####
################################################################

def _cycle_indices(G, ci, order_fcn = None):
	"""
		Return matrix Psi_i s.t.
			Psi_i alpha_i
		is a vector with respect to ordering
	"""
	if order_fcn == None:
		nodelist = G.nodes()
		order_fcn = lambda v : nodelist.index(v)
	row_idx = [order_fcn(v) for v in ci]
	col_idx = range(len(ci))
	vals = np.ones(len(ci))
	return scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape = (len(G), len(ci)) )

def _stateind(G, node, mode, order_fcn = None):
	'''
	For a given 'node' and 'mode' in the graph 'G', compute the index
	of the node-mode state in the corresponding linear system, using
	ordering given by order_fcn
	'''
	if order_fcn == None:
		nodelist = G.nodes()
		order_fcn = lambda v : nodelist.index(v)
	return (mode - 1) * len(G) + order_fcn(node)

def _cycle_matrix(G, cycle, mode):
	'''
	Comput a square matrix with rows given by _cycle_rows
	'''
	return [row for row in _cycle_rows(G, cycle, mode)]

def _cycle_rows(G, C, mode):
	'''
	Return iterator that produces all combinations of 0/1 vectors vi for a given cycle C in a graph G,
	such that vi[j] = 1 if   the mode of the edge (i + j mod L, i + j + 1 mod L) is 'mode' 
	'''
	d = deque()
	for i in range(len(C)):
		v1 = C[i]
		v2 = C[(i+1) % len(C)]
		d.append(1 if G[v1][v2]['mode'] == mode else 0)
	for i in range(len(C)):
		yield list(d)
		d.rotate(-1)

def _maxmode(G):
	'''
	For a given graph 'G', find the largest mode integer.
	'''
	return max([d['mode'] for (u,v,d) in  G.edges(data=True)])

################################################################
##### Functions used to round non-integer assignments ##########
################################################################

def casc_round(list):
	'''
		Round entries in list while preserving total sum:
		i_n = round(f_0 + ... + f_n) - (i_0 + ... + i_n-1)
	'''
	fp_total = 0.
	int_total = 0

	ret = [0] * len(list)

	for i,fp in enumerate(list):
		ret[i] = round(fp_total + fp) - int_total
		fp_total += fp
		int_total += ret[i]

	return ret

def make_integer(assignments):
	'''
	Round assignments in 'assignments' using cascade rounding s.t.
	 - sum of all assignments is preserved
	 - every assignment is integral
	'''
	ass_sums = [sum(a) for a in assignments]
	rounded_sums = casc_round(ass_sums)

	for ass, int_sum in zip(assignments, rounded_sums):
		diff = int_sum - sum(ass)
		for j in range(len(ass)):
			ass[j] = ass[j] + diff/len(ass)

	return [casc_round(ass) for ass in assignments]

def make_avg_integer(assignments):
	'''
	Round assignments in 'assignments' by first constructing average assignments s.t.
	 - sum of all assignments is preserved
	 - every assignment is integral
	'''
	ass_sums = [sum(a) for a in assignments]			# sum of assignments
	rounded_sums = casc_round(ass_sums) 				# round assignment sums to integers

	# create averaged assignments
	avg_assignments = [ [rounded_sums[i]/len(assignments[i])] * len(assignments[i]) for i in range(len(assignments)) ]

	return [casc_round(ass) for ass in avg_assignments]
