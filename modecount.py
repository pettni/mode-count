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

from mosek.fusion import *
from mosek.array import *

from cvxopt import matrix, spmatrix, solvers

import itertools

from random_cycle import random_cycle

class Abstraction(object):
	""" 
		Abstraction built on top of a networkx DiGraph
	"""
	def __init__(self, lb, ub, eta, tau):
		self.eta = eta
		self.tau = float(tau)

		self.lb = np.array(lb, dtype=np.float64)
		# self.lb -= (self.lb + float(eta)/2) % eta  # make sure that zero is in middle of cell
		self.ub = np.array(ub, dtype=np.float64)
		self.ub += (self.ub - self.lb) % eta # make domain number of eta's
		self.nextMode = 1

		# number of discrete points along each dimension
		self.n_dim = np.round((self.ub - self.lb)/eta).astype(int)

		# Create a graph and populate it with nodes
		self.graph = nx.DiGraph()
		for midx in itertools.product(*[range(dim) for dim in self.n_dim]):
			cell_lb = self.lb + np.array(midx) * eta
			cell_ub = cell_lb + eta
			self.graph.add_node(midx, lb = tuple(cell_lb), ub = tuple(cell_ub), mid = (cell_lb + cell_ub) / 2)

	def __len__(self):
		return len(self.graph)

	def get_midx_pt(self, pt):
		""" 
			Return the discrete multiindex corresponding to a given continuous point
		"""
		assert(len(pt) == len(self.n_dim))
		if not self.contains(pt):
			raise('Point outside domain')
		midx = np.floor((np.array(pt) - self.lb) / self.eta).astype(np.uint64)
		return tuple(midx)

	def contains(self, pt):
		if np.any(self.lb >= pt) or np.any(self.ub <= pt):
			return False
		return True

	def plot_planar(self):
		"""
			In 2d case, plot abstraction with transitions
		"""
		assert(len(self.lb) == 2)
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
		"""
			Add new dynamic mode to the abstraction, given by
			the vector field vf 
		"""
		dummy_vf = lambda z,t : vf(z)
		tr_out = 0
		for node, attr in self.graph.nodes_iter(data=True):
			x_fin = scipy.integrate.odeint(dummy_vf, attr['mid'], np.arange(0, self.tau, self.tau/100))
			if self.contains(x_fin[-1]):
				midx = self.get_midx_pt(x_fin[-1])
				self.graph.add_edge( node, midx, mode = self.nextMode )
			else:
				tr_out += 1
		if tr_out > 0:
			print "Warning: ", tr_out, " transitions out of ", len(self.graph), " in mode ", self.nextMode, " go out of domain"
		self.nextMode += 1

	def verify_bisim(self, beta, eps):
		return (beta(eps, self.tau) + self.eta/2 <= eps)

	def node_to_idx(self, node):
		# Given a node (x1,x2,x3, ...),
		# return the list index on the form
		#
		#   Lz ( Ly (x) + y ) + z
		# 
		assert len(node) == len(self.n_dim)
		ret = node[0]
		for i in range(1,len(self.n_dim)):
			ret *= self.n_dim[i]
			ret += node[i]
		return ret

	def idx_to_node(self, idx):
		# Inverse to node_to_idx
		assert(idx < np.product(self.n_dim))
		node = [0] * len(self.n_dim)
		for i in reversed(range(len(self.n_dim))):
			node[i] = int(idx % self.n_dim[i])
			idx = np.floor(idx / self.n_dim[i])
		return tuple(node)

class CycleControl():
	def __init__(self, G, sol, order_fcn):
		self.G = G
		self.u_list = sol['controls']
		self.c_list = sol['cycles']
		self.alpha_list = [deque(a) for a in sol['assignments']]
		self.order_fcn = order_fcn
		self.A, self.B = lin_syst(G, order_fcn)
		self.stacked_e = _stacked_eye(G)

	def get_u(self, t, state):
		
		if t < len(self.u_list):
			return np.array(self.u_list[t])

		# compute desired next state
		for a in self.alpha_list: a.rotate()
		
		# solve linear system to find control
		sum_next = np.sum([_cycle_indices(self.G, c, self.order_fcn).dot( a ) for c,a in zip(self.c_list, self.alpha_list)], 0)
		b_lin = sum_next - self.stacked_e.dot(self.A.dot( state ) )
		A_lin = self.stacked_e.dot( self.B )
	
		M = Model()
		u = M.variable(self.B.shape[1], Domain.greaterThan(0.))
		M.constraint( Expr.sub(state, u), Domain.greaterThan(0.))
		M.constraint( Expr.sub( Expr.mul(_sparse_scipy_to_mosek(A_lin), u), list(b_lin)), Domain.equalsTo(0.) )

		M.acceptedSolutionStatus(AccSolutionStatus.Anything)

		M.solve()
		
		return np.array(u.level())

def draw_modes(G):
	""" 
		Given a DiGraph G, w

	"""
	maxmode = max([d['mode'] for (u,v,d) in  G.edges(data=True)]) + 1
	edgelists = [ [(u,v) for (u,v,d) in G.edges(data=True) if d['mode'] == mode ]  for mode in range(1, maxmode)]
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(edgelists) + 1)))
	
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_color = [next(colors)] * G.number_of_nodes() )
	for i, edgelist in enumerate(edgelists):
		nx.draw_networkx_edges(G, pos, edgelist = edgelist, edge_color = [next(colors)] * len(edgelist))
	nx.draw_networkx_labels(G,pos)

def lin_syst(G, order_fcn = None):
	""" 
		Return matrices A,B such that
		[w_1(t+1); w_2(t+1); ...] = A [w_1(t); w_2(t); ...] + B [r_1(t); r_2(t); ...]
		for modes 1, 2 ...
	"""
	if order_fcn == None:
		order_fcn = lambda v : G.nodes().index(v)

	ordering = sorted(G.nodes_iter(), key = order_fcn)

	T = nx.adjacency_matrix(G, nodelist = ordering, weight='mode').transpose()
	T_list = [ T == mode for mode in range(1, _maxmode(G)+1) ]

	A = scipy.sparse.block_diag(tuple(T_list), dtype=np.int8)	
	B = scipy.sparse.bmat([ [Ti for i in range(len(T_list))] for Ti in T_list ]) - 2*A

	return A,B

def _dyn_eq_mat(A, B, T):
	# compute matrix Aeq s.t.
	#  x(t+1) = Ax + Bu 
	# for t = 0, ..., T-1
	N = A.shape[1]
	m = B.shape[1]

	N_eq_dyn = N * T

	A_dyn_u = scipy.sparse.block_diag((B,) * T)
	A_dyn_x = scipy.sparse.bmat( [[scipy.sparse.block_diag((A,) * T), scipy.sparse.coo_matrix( (N_eq_dyn, N) ) ]]) \
		+  scipy.sparse.bmat( [[scipy.sparse.coo_matrix( (N_eq_dyn, N) ), -scipy.sparse.identity( N_eq_dyn ) ]]) \

	return scipy.sparse.bmat([[A_dyn_u, A_dyn_x]])

def _verify_dims(list, hl, tot):
	assert( np.sum([ mat.shape[1] for mat in list ]) == tot )
	assert( np.all([ mat.shape[0] == len(hl) for mat in list ] ))

def _coo_zeros(i,j):
	return scipy.sparse.coo_matrix( (i,j) )

def _sparce_scipy_to_cvxopt(A):
	A_coo = A.tocoo()
	return spmatrix(A_coo.data.astype(float), A_coo.row.astype(int), A_coo.col.astype(int), (A_coo.shape[0], A_coo.shape[1]))

def synthesize2(G, init, T, K, mode, cycle_set = [], forbidden_nodes = [], order_fcn = None, integer = True, verbosity = 0):

	if order_fcn == None:
		nodelist = G.nodes()
		order_fcn = lambda v : nodelist.index(v)

	A,B = lin_syst(G, order_fcn)
	maxmode = _maxmode(G)
	N = A.shape[1]

	assert(len(G) == N / maxmode)

	# fill out cycle set
	if len(cycle_set) == 0:

		if False:
			# find ranom cycles
			while len(cycle_set) < 100:
				cycle_set.append(random_cycle(G))
		else:
			# enumerate all simple cycles
			for c in nx.simple_cycles(G):
				if not set(forbidden_nodes) & set(c):
					# only care about cycles not involving roots
					cycle_set.append(c)

	# optimization vector is 
	# 
	#  [ u[0], ..., u[T-1], x[0], ..., x[T], a[0], ..., a[C-1], Kl[1], ..., Kl[C], Ku[1], ..., Ku[C], Kl, Ku, err ]
	#

	N_cycle = [len(cycle) for cycle in cycle_set]  	# length of cycles

	# variable block dimensions
	N_u 	= T * N 			# input vars
	N_x     = (T+1) * N 		# state vars
	N_cycle_tot = sum(N_cycle)	# cycle vars
	N_lb 	= len(cycle_set)	# lb vars
	N_ub 	= len(cycle_set)	# ub vars
	N_lb_tot = 1
	N_ub_tot = 1
	N_err = 1

	N_tot = N_u + N_x + N_cycle_tot + N_lb + N_ub + N_lb_tot + N_ub_tot + N_err

	# number of equalities
	N_eq_dyn = N * T 	  # dynamics
	N_eq_0 = len(G)  # initial state agreement
	N_eq_T = len(G)  # time T constraint
	N_eq_forbidden = (T+1) * maxmode * len(forbidden_nodes) # forbidden nodes constraint

	N_ineq_mc_trans = T+1 	 					# transient mode constraint
	N_ineq_control = N_u 						# upper/lower ctrl bounds
	N_ineq_cycle_lb = sum(N_cycle) 				# upper bounds on cycle
	N_ineq_cycle_ub = sum(N_cycle) 				# upper bounds on cycle
	N_ineq_tot_lb = 1
	N_ineq_tot_ub = 1
	N_ineq_err = 2

	index_ut =  lambda t : t * N 			# starting index for u[t], t = 0, ..., T-1
	index_xt =  lambda t : N_u + t * N 	# starting index for x[t], t = 0, ..., T
	index_ai =  lambda i : N_u + N_x + sum([N_cycle[ii] for ii in range(i) ])  	# starting index for a[i], i = 0, ..., C-1
	index_Kli = lambda i : N_u + N_x + N_cycle_tot + i 							# index of Kl[i]
	index_Kui = lambda i : N_u + N_x + N_cycle_tot + N_lb + i 					# index of Ku[i] 
	index_Kl = N_u + N_x + N_cycle_tot + N_lb + N_ub 							# index of Kl
	index_Ku = N_u + N_x + N_cycle_tot + N_lb + N_ub  + N_ub_tot				# index of Ku
	index_err = N_u + N_x + N_cycle_tot + N_lb + N_ub  + N_ub_tot + N_lb_tot	# index of err

	# assert this stuff makes sense
	assert(index_err == N_tot - 1)
	assert(index_ut(T-1) + N == index_xt(0) )
	assert(index_xt(T) + N == index_ai(0) )
	assert(index_ai(len(cycle_set) - 1) + len(cycle_set[-1]) == index_Kli(0) )
	assert(index_Kl == index_Kui(len(cycle_set) - 1) + 1)

	# Compute cycle -> vertex matrices for all cycles
	Psi_mats = []    # matrices cycle -> state
	for cycle in cycle_set:
		Psi = _cycle_indices(G, cycle, order_fcn)
		assert(Psi.shape[0] == len(G))
		assert(Psi.shape[1] == len(cycle))
		Psi_mats.append( Psi )


	# Build equality constraints Aeq x = beq

	# dynamics constraints
	Aeq11 = _dyn_eq_mat(A,B,T)
	Aeq12 = scipy.sparse.coo_matrix( (N_eq_dyn, N_cycle_tot + N_lb + N_ub  + N_lb_tot + N_ub_tot + N_err) )
	beq1 = np.zeros( N_eq_dyn )

	_verify_dims([Aeq11, Aeq12], beq1, N_tot)

	# initial constraints
	Aeq21 = scipy.sparse.coo_matrix( (N_eq_0, N_u) )						 	# zeros
	Aeq22 = _stacked_eye(G)
	Aeq23 = scipy.sparse.coo_matrix( (N_eq_0,  N * T + N_cycle_tot + N_lb + N_ub + N_lb_tot + N_ub_tot + N_err) )				# zeros
	beq2 = np.array(init)

	_verify_dims([Aeq21, Aeq22, Aeq23], beq2, N_tot)

	# final constraints
	Aeq31 = scipy.sparse.coo_matrix( (N_eq_T, N_u + N*T) )
	Aeq32 = _stacked_eye(G)
	Aeq33 = scipy.sparse.bmat([ Psi_mats ])
	Aeq34 = scipy.sparse.coo_matrix( (N_eq_T, N_lb + N_ub  + N_ub_tot + N_lb_tot + N_err) )
	beq3 = np.zeros(N_eq_T)

	_verify_dims([Aeq31, Aeq32, Aeq33, Aeq34], beq3, N_tot)

	# forbidden node constraints
	Aeq4 = scipy.sparse.coo_matrix( (np.ones(N_eq_forbidden), 
		( range(N_eq_forbidden) , 
		 [ N_u + t * N + _stateind(G, f_node, m, order_fcn) for f_node in forbidden_nodes for t in range(T+1) for m in range(1, maxmode+1)  ]  
		) ),
		 (N_eq_forbidden, N_tot) )
	beq4 = np.zeros(N_eq_forbidden)

	_verify_dims([Aeq4], beq4, N_tot)
	
	Aeq1 = scipy.sparse.bmat([[ Aeq11, Aeq12 ]])
	Aeq2 = scipy.sparse.bmat([[ Aeq21, Aeq22, Aeq23 ]]) 
	Aeq3 = scipy.sparse.bmat([[ Aeq31, Aeq32, Aeq33, Aeq34]]) 

	Aeq = scipy.sparse.bmat([[ Aeq1], [Aeq2], [Aeq3], [Aeq4]])
	beq = np.hstack([beq1, beq2, beq3, beq4])

	# Build inequality constraints Aiq x <= biq
	# N_ineq_mc_trans = T 	 					# transient mode constraint
	# N_ineq_control = 2 * N_u 					# upper/lower ctrl bounds
	# N_ineq_cycle_lb = sum(N_cycle) 				# upper bounds on cycle
	# N_ineq_cycle_ub = sum(N_cycle) 				# upper bounds on cycle
	# N_ineq_tot_lb = 1
	# N_ineq_tot_ub = 1
	# N_ineq_err = 2

	sum_mode_mat = scipy.sparse.coo_matrix( ( np.ones(len(G)), ( np.zeros(len(G)), [(mode - 1)*len(G) + i for i in range(len(G))] ) ), 
									(1, N) )

	# lower bound mode counting
	Aiq11 = _coo_zeros( N_ineq_mc_trans, N_u)
	Aiq12 = scipy.sparse.block_diag( (-sum_mode_mat,) * (T+1) )
	Aiq13 = _coo_zeros( N_ineq_mc_trans, N_cycle_tot + N_lb + N_ub)
	Aiq14 = scipy.sparse.coo_matrix( (np.ones(T+1), ( range(T+1), np.zeros(T+1) )  ), (T+1, 1) )
	Aiq15 = _coo_zeros( N_ineq_mc_trans, N_ub_tot + N_err)
	biq1 = np.zeros(N_ineq_mc_trans)

	_verify_dims([Aiq11, Aiq12, Aiq13, Aiq14, Aiq15], biq1, N_tot)

	# upper bound mode counting
	Aiq21 = Aiq11
	Aiq22 = scipy.sparse.block_diag( (sum_mode_mat,) * (T+1) )
	Aiq23 = _coo_zeros( N_ineq_mc_trans, N_cycle_tot + N_lb + N_ub + N_lb_tot)
	Aiq24 = scipy.sparse.coo_matrix( (-np.ones(T+1), ( range(T+1), np.zeros(T+1) )  ), (T+1, 1) )
	Aiq25 = _coo_zeros( N_ineq_mc_trans, N_err)
	biq2 = np.zeros(N_ineq_mc_trans)

	_verify_dims([Aiq21, Aiq22, Aiq23, Aiq24, Aiq25], biq2, N_tot)

	# lower control bounds
	Aiq31 = -scipy.sparse.identity( N_u )
	Aiq32 = _coo_zeros( N_ineq_control, N_tot - N_u )
	biq3 = np.zeros(N_ineq_control)

	_verify_dims([Aiq31, Aiq32], biq3, N_tot)

	# upper control bounds
	Aiq41 = scipy.sparse.identity(N_ineq_control)
	Aiq42 = -scipy.sparse.identity(N_ineq_control)
	Aiq43 = _coo_zeros(N_ineq_control, N + N_cycle_tot + N_lb + N_ub + N_lb_tot + N_ub_tot + N_err)
	biq4 = np.zeros(N_ineq_control)

	_verify_dims([Aiq41, Aiq42, Aiq43], biq4, N_tot)

	# lower cycle bounds
	Aiq51 = _coo_zeros( N_ineq_cycle_lb, N_u + N_x )
	Aiq52 = -scipy.sparse.block_diag( tuple( [ _cycle_matrix( G,c,mode ) for c in cycle_set ] ) )
	Aiq53 = scipy.sparse.block_diag( tuple( [ np.ones([len(c), 1]) for c in cycle_set ] ) )
	Aiq54 = _coo_zeros( N_ineq_cycle_lb, N_ub + N_lb_tot + N_ub_tot + N_err )
	biq5 = np.zeros(N_ineq_cycle_lb)

	_verify_dims([Aiq51, Aiq52, Aiq53, Aiq54], biq5, N_tot)

	# upper cycle bounds
	Aiq61 = _coo_zeros( N_ineq_cycle_lb, N_u + N_x )
	Aiq62 = scipy.sparse.block_diag( tuple( [ _cycle_matrix( G,c,mode ) for c in cycle_set ] ) )
	Aiq63 = _coo_zeros( N_ineq_cycle_ub, N_lb )
	Aiq64 = -scipy.sparse.block_diag( tuple( [ np.ones([len(c), 1]) for c in cycle_set ] ) )
	Aiq65 = _coo_zeros( N_ineq_cycle_lb, N_lb_tot + N_ub_tot + N_err )
	biq6 = np.zeros(N_ineq_cycle_lb)

	_verify_dims([Aiq61, Aiq62, Aiq63, Aiq64, Aiq65], biq6, N_tot)

	# set sum(ub) < ub_tot, lb_tot < sum(lb)
	Aiq71 = _coo_zeros(N_lb_tot + N_ub_tot, N_u + N_x + N_cycle_tot)
	Aiq72 = scipy.sparse.block_diag( (-np.ones([1, N_lb]), np.ones([1, N_ub])) )
	Aiq73 = scipy.sparse.identity(N_lb_tot + N_ub_tot)
	Aiq74 = _coo_zeros(N_lb_tot + N_ub_tot, N_err)
	biq7  = np.zeros(N_lb_tot + N_ub_tot)

	_verify_dims([Aiq71, Aiq72, Aiq73, Aiq74], biq7, N_tot)

	# set err > ub - L,  err > K - lb
	Aiq81 = _coo_zeros(2, N_u + N_x + N_cycle_tot + N_lb + N_ub)
	Aiq82 = np.array([[ -1, 0, -1 ], [0, 1, -1]])
	biq8  = np.array([ K, -K ])

	_verify_dims([Aiq81, Aiq82], biq8, N_tot)

	Aiq1 = scipy.sparse.bmat([ [ Aiq11, Aiq12, Aiq13, Aiq14, Aiq15 ]] )
	Aiq2 = scipy.sparse.bmat([ [ Aiq21, Aiq22, Aiq23, Aiq24, Aiq25 ]] )
	Aiq3 = scipy.sparse.bmat([ [ Aiq31, Aiq32 ]] )
	Aiq4 = scipy.sparse.bmat([ [ Aiq41, Aiq42, Aiq43 ]] ) 
	Aiq5 = scipy.sparse.bmat([ [ Aiq51, Aiq52, Aiq53, Aiq54 ]] )
	Aiq6 = scipy.sparse.bmat([ [ Aiq61, Aiq62, Aiq63, Aiq64, Aiq65 ]] )
	Aiq7 = scipy.sparse.bmat([ [ Aiq71, Aiq72, Aiq73, Aiq74 ]] )
	Aiq8 = scipy.sparse.bmat([ [ Aiq81, Aiq82 ]] )

	Aiq = scipy.sparse.bmat([ [Aiq1], [Aiq2], [Aiq3], [Aiq4], [Aiq5], [Aiq6], [Aiq7], [Aiq8] ])
	biq = np.hstack([biq1, biq2, biq3, biq4, biq5, biq6, biq7, biq8])

	# cost function
	c = np.zeros(N_tot)
	c[-1] = 1

	sol = solvers.lp(matrix(c), _sparce_scipy_to_cvxopt(Aiq), matrix(biq), _sparce_scipy_to_cvxopt(Aeq), matrix(beq), 'mosek')

	x_out = sol['x']

	final_c = []
	final_a = []
	for i in range(len(cycle_set)):
		alpha_i = np.array(x_out[index_ai(i): index_ai(i) + len(cycle_set[i])])
		if sum( alpha_i ) > 1e-5:
			if verbosity:
				print "Cycle: ", cycle_set[i]
				print "Cycle mode-count: ", next(_cycle_rows(G,cycle_set[i], mode))
				print "Assignment: ", alpha_i
			final_c.append(cycle_set[i])
			final_a.append(alpha_i)

	# for t in range(1,T+1):
	# 	# check that sol obeys dynamics up to time T
	# 	if (np.all(np.abs(A.dot(x_t[t-1].level()) + B.dot(u_t[t-1].level()) - x_t[t].level()) > 1e-10)):
	# 		print "Warning, dynamics mismatch for ", np.max(np.abs(A.dot(x_t[t-1].level()) + B.dot(u_t[t-1].level()) - x_t[t].level()))

	sol['controls'] = np.array([np.array(x_out[t * N : (t+1) * N ]) for t in range(T)])
	sol['states'] = np.array([np.array(x_out[N_u + t * N : N_u + (t+1) * N ]) for t in range(T+1)])
	sol['cycles'] = final_c
	sol['assignments'] = final_a
	sol['forbidden_nodes'] = forbidden_nodes

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

	init = sol['states'][0]
	xvec = np.zeros([len(init), tmax+1], dtype=float)
	uvec = np.zeros([len(init), tmax+1], dtype=float)
	xvec[:,0] = init.flatten()
	for t in range(tmax):
		u = controls.get_u(t, xvec[:,t])
		uvec[:,t] = u.flatten()
		xvec[:,t+1] = A.dot(xvec[:,t]) + B.dot(u.flatten())

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
	for node in sol['forbidden_nodes']:
		if node in subgraph.nodes():
			node_size[subgraph.nodes().index(node)] = 600

	# nx.draw_networkx_labels(subgraph, pos, ax=ax)
	
	nod = nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size = node_size )

	edge_anim = [ (v, w, plt.plot(pos[v], pos[w], marker='o', color = 'w', markersize=10) ) for v,w in subgraph.edges_iter()]

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

def _stacked_eye(G):
	n_v = len(G)
	n = len(G) * _maxmode(G)
	return scipy.sparse.coo_matrix( (np.ones(n), ([ (i % n_v ) for i in range(n)], range(n) )), shape = (n_v, n) )

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

def _sparse_scipy_to_mosek(A):
	A_coo = A.tocoo()
	return Matrix.sparse(A_coo.shape[0], A_coo.shape[1], list(A_coo.row.astype(int)), list(A_coo.col.astype(int)), list(A_coo.data.astype(float)))

def _maxmode(G):
	return max([d['mode'] for (u,v,d) in  G.edges(data=True)])

def _stateind(G, node, mode, order_fcn = None):
	if order_fcn == None:
		nodelist = G.nodes()
		order_fcn = lambda v : nodelist.index(v)
	return (mode - 1) * len(G) + order_fcn(node)

def _cycle_matrix(G, cycle, mode):
	return [row for row in _cycle_rows(G, cycle, mode)]

def _cycle_rows(G, C, mode):
	"""
		Return iterator that produces all combinations of 0/1 vectors vi for a given cycle C in a graph G,
		such that vi[j] = 1 if   the mode of the edge (i + j mod L, i + j + 1 mod L) is `mode' 
	"""
	d = deque()
	for i in range(len(C)):
		v1 = C[i]
		v2 = C[(i+1) % len(C)]
		d.append(1. if G[v1][v2]['mode'] == mode else 0.)
	for i in range(len(C)):
		yield list(d)
		d.rotate(-1)

def _sum_modes(state, maxmode):
	return np.sum([state[ i * len(state)/maxmode : (i+1) * len(state)/maxmode] for i in range(maxmode)], axis=0)

def _psi_lower(G, cycle, assignment, mode):
	return np.min([ np.dot(assignment, cr) for cr in _cycle_rows(G, cycle, mode)])

def _psi_upper(G, cycle, assignment, mode):
	return np.max([ np.dot(assignment, cr) for cr in _cycle_rows(G, cycle, mode)])
