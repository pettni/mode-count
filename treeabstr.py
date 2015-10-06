from collections import deque

import numpy as np

from numpy.linalg import norm
from scipy.linalg import expm
import scipy.integrate
import scipy.sparse
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx

from mosek.fusion import *
from mosek.array import *

import itertools

def draw_modes(G):
	maxmode = max([d['mode'] for (u,v,d) in  G.edges(data=True)]) + 1
	edgelists = [ [(u,v) for (u,v,d) in G.edges(data=True) if d['mode'] == mode ]  for mode in range(1, maxmode)]
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(edgelists) + 1)))
	
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_color = [next(colors)] * G.number_of_nodes() )
	for i, edgelist in enumerate(edgelists):
		nx.draw_networkx_edges(G, pos, edgelist = edgelist, edge_color = [next(colors)] * len(edgelist))
	nx.draw_networkx_labels(G,pos)

def lin_syst(G):
	""" 
		Return matrices A,B such that
		[w_1(t+1); w_2(t+1); ...] = A [w_1(t); w_2(t); ...] + B [r_1(t); r_2(t); ...]
		for modes 1, 2 ...
	"""
	maxmode = max([d['mode'] for (u,v,d) in  G.edges(data=True)])
	T = nx.adjacency_matrix(G, weight='mode').transpose()
	T_list = [ T == mode for mode in range(1, maxmode+1) ]

	A = scipy.sparse.block_diag(tuple(T_list), dtype=np.int8)	
	B = scipy.sparse.bmat([ [Ti for i in range(len(T_list))] for Ti in T_list ]) - 2*A

	return A,B

def solve_lp(G, init):

	def sparse_scipy_to_mosek(A):
		A_coo = A.tocoo()
		return Matrix.sparse(A_coo.shape[0], A_coo.shape[1], list(A_coo.row.astype(int)), list(A_coo.col.astype(int)), list(A_coo.data.astype(float)))

	def cycle_indices(ci):
		"""
			Return matrix Psi_i s.t.
				Psi_i alpha_i
			is a vector with respect to G.nodes() 
		"""
		row_idx = [G.nodes().index(v) for v in ci]
		col_idx = range(len(ci))
		vals = np.ones(len(ci))
		return Matrix.sparse(len(G), len(ci), row_idx, col_idx, vals)

	T = 10
	K = 50/4.

	mode = 1  # mode to bound

	A,B = lin_syst(G)

	n = A.shape[1]
	m = B.shape[1]
	n_v = len(G)

	maxmode = n / n_v

	tree_roots = np.nonzero(A.diagonal())[0] # index of nonzero elements
	tree_roots_graph = tree_roots - np.array([i * len(G) for i in range(maxmode)])
	tree_roots_nodes = [G.nodes()[k] for k in tree_roots_graph]

	A_mos = sparse_scipy_to_mosek(A)
	B_mos = sparse_scipy_to_mosek(B)

	stacked_eye = Matrix.sparse( n_v, n, [ (i % n_v ) for i in range(n)], range(n), np.ones(n))

	# Compute graph cycles
	with Model() as M: 

		#########################################
		######### Define variables ##############
		#########################################

		# control variables u(0), ... u(T-1)
		u_t = [None] * (T)
		for t in range(T):
			u_t[t] = M.variable("u[" + str(t) + "]", m, Domain.greaterThan(0.0)) 

		# state variables x(0), ... x(T)
		x_t = [None] * (T+1)
		for t in range(0, T+1):
			x_t[t] = M.variable("x[" + str(t) + "]", m, Domain.greaterThan(0.0)) 

		# cycle variables	
		c_i = []
		alpha_i = []  # cycle assignments
		Psi_i = []    # matrices cycle -> state
		k_u_i = []    # upper mode counting bounds
		k_l_i = []	  # lower mode counting bounds
		for i, c in enumerate(nx.simple_cycles(G)):
			if not set(tree_roots_nodes) & set(c):
				# only care about cycles not involving roots
				print next(cycle_rows(G,c, mode))
				c_i.append(c)
				alpha_i.append(M.variable("alpha_" + str(i), len(c), Domain.greaterThan(0.0)))
				Psi_i.append(cycle_indices(c))
				k_u_i.append(M.variable( 1, Domain.unbounded() ))
				k_l_i.append(M.variable( 1, Domain.unbounded() ))

		# error variables
		err = M.variable("error", 1, Domain.greaterThan(0.))

		#########################################
		######### Define constraints ############
		#########################################

		# Constraint at t=0
		M.constraint( Expr.mul(stacked_eye, x_t[0]), Domain.equalsTo(init))
		
		# Dynamics constraints
		for t in range(1, T+1):
			M.constraint( Expr.sub(x_t[t], Expr.add(Expr.mul(A_mos, x_t[t-1]), Expr.mul(B_mos, u_t[t-1])) ), Domain.equalsTo(0.))
			# TODO: TRANSIENT CONSTRAINTS

		# Control constraints
		for t in range(T):
			M.constraint( Expr.sub(x_t[t], u_t[t]), Domain.greaterThan(0.) )

		# Equality at time T
		Psi_stacked = Matrix.sparse([Psi_i])
		alpha_stacked = Expr.vstack([alpha.asExpr() for alpha in alpha_i])
		M.constraint( Expr.sub(Expr.mul(Psi_stacked, alpha_stacked), Expr.mul(stacked_eye, x_t[T])), Domain.equalsTo(0.) )

		# Bound assignments in [k_l_i[i], k_u_i[i]]
		for i, c in enumerate(c_i):
			Ac = DenseMatrix(np.array(cycle_matrix(G, c, mode), dtype=float) )
			M.constraint( Expr.sub(Expr.mul(Ac, alpha_i[i]), Expr.mul(k_l_i[i], [1. for j in range(len(c))]) ), Domain.greaterThan(0.) )
			M.constraint( Expr.sub(Expr.mul(k_u_i[i], [1. for j in range(len(c))]), Expr.mul(Ac, alpha_i[i]) ), Domain.greaterThan(0.) )

		# Set \sum k_l_i[i] <= K <= \sum k_u_i[i] 
		k_u_sum = Expr.sum(  Expr.vstack([k.asExpr() for k in k_u_i]) )
		k_l_sum = Expr.sum(  Expr.vstack([k.asExpr() for k in k_l_i]) )
 		M.constraint( Expr.sub(err, Expr.sub( k_u_sum, K ) ), Domain.greaterThan(0.) )
 		M.constraint( Expr.sub(err, Expr.sub( K, k_l_sum ) ), Domain.greaterThan(0.) )

		M.objective(ObjectiveSense.Minimize, err)

		# Enable logger output
		M.setLogHandler(sys.stdout) 

		M.solve()

		print "Primal solution status: ", M.getPrimalSolutionStatus()  
		print "Dual solution status: ", M.getDualSolutionStatus()  

		print "Error: ", err.level()[0]
		print "Guaranteed interval: [", K-err.level()[0], ", ", K+err.level()[0], "]"
		print ""
		print "Cycles:"
		for i, alpha in enumerate(alpha_i):
			if sum(alpha.level()) > 1e-5:
				print "Cycle mode-count: ", next(cycle_rows(G,c_i[i], mode))
				print "Cycle length: ", len(alpha.level())
				print "Assignment: ", alpha.level()

		conn_com = max(nx.strongly_connected_components(G), key=len)
		conn_com_ind = [G.nodes().index(i) for i in conn_com]
		for t in range(T+1):
			print sum(x_t[t].level())
			print sum_modes(x_t[t].level(), 2)[conn_com_ind]

def cycle_matrix(G, cycle, mode):
	return [row for row in cycle_rows(G, cycle, mode)]

def cycle_rows(G, C, mode):
	"""
		Return iterator that produces all combinations of 0/1 vectors vi for a given cycle C in a graph G,
		such that vi[j] = 1 if   the mode of the edge (i + j mod L, i + j + 1 mod L) is `mode' 
	"""
	d = deque()
	for i in range(len(C)):
		v1 = C[i]
		v2 = C[(i+1) % len(C)]
		d.append(1 if G[v1][v2]['mode'] == mode else 0)
	for i in range(len(C)):
		yield list(d)
		d.rotate(-1)

def midx_to_idx(midx, mN):
	# Given a len(mN)-dimensional matrix of with sides mN stored as a list,
	# return the list index corresponding to the multiindex midx
	#
	# For mN = [Lx, Ly, Lz], midx = [ix, iy, iz]
	#   idx = Lz ( Ly (x) + y ) + z
	# 
	assert len(midx) == len(mN)
	if len(midx) == 1:
		return int(midx[0])
	else:
		return int(midx[-1] + mN[-1] * midx_to_idx(midx[0:-1], mN[0:-1]))

def idx_to_midx(idx, mN):
	# Inverse to midx_to_idx
	midx = [0] * len(mN)
	for i in reversed(range(len(mN))):
		midx[i] = int(idx % mN[i])
		idx = np.floor(idx / mN[i])
	return tuple(midx)

class Abstraction(object):
	""" 
		Abstraction built on top of a networkx DiGraph
	"""
	def __init__(self, lb, ub, eta):
		self.eta = eta

		self.lb = np.array(lb, dtype=np.float64)
		self.lb -= (self.lb + float(eta)/2) % eta  # make sure that zero is in middle of cell
		self.ub = np.array(ub, dtype=np.float64)
		self.ub += (self.ub - self.lb) % eta # make domain number of eta's
		self.nextMode = 1

		# number of discrete points along each dimension
		self.n_dim = np.round((self.ub - self.lb)/eta).astype(np.uint64)

		# Create a graph and populate it with nodes
		self.graph = nx.DiGraph()
		for midx in itertools.product(*[range(dim) for dim in self.n_dim]):
			cell_lb = self.lb + np.array(midx) * eta
			cell_ub = cell_lb + eta
			self.graph.add_node(midx, lb = tuple(cell_lb), ub = tuple(cell_ub), mid = (cell_lb + cell_ub) / 2)

	def get_midx_pt(self, pt):
		""" 
			Return the discret multiindex corresponding to a given continuous point
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

	def add_mode(self, vf, tau):
		"""
			Add new dynamic mode to the abstraction, given by
			the vector field vf and discretization tau
		"""
		dummy_vf = lambda z,t : vf(z)
		for node, attr in self.graph.nodes_iter(data=True):
			x_fin = scipy.integrate.odeint(dummy_vf, attr['mid'], np.arange(0, tau, tau/100))
			if self.contains(x_fin[-1]):
				midx = self.get_midx_pt(x_fin[-1])
				self.graph.add_edge( node, midx, mode = self.nextMode )
			else:
				print "Warning: transition from ", attr['mid'], " in mode ", self.nextMode, " goes out of domain"
		self.nextMode += 1

def verify_bisim(data):
	for beta in data['beta']: 
		assert(beta(data['eps'], data['tau']) + data['eta']/2 <= data['eps'])

def verify_tree(data):
	for beta in data['beta']: 
		for s in np.arange(0, 2*data['eta'], 0.02):
			try:
				assert(beta(s, data['tau']) <= np.maximum(data['eta']/2, s - data['eta']/2))
			except Exception, e:
				print "tree verif failed for s = ", s

def plot_treeineq(data):
	s = np.arange(0, 3*data['eta'], 0.01)
	c = np.maximum(data['eta']/2, s - data['eta']/2)
	plt.plot(s,c, 'b')
	for beta in data['beta']:
		b = beta(s, data['tau'])
		plt.plot(s,b, 'r')
	plt.show()

def tree_abstr(data):
	""" 
		Compute an abstraction
	"""
	ab = Abstraction(data['lb'], data['ub'], data['eta'])

	for vf in data['vf']:
		ab.add_mode(vf, data['tau'])
	return ab

def example_abstr():
	data = {}

	# Define a vector fields
	data['vf'] = [lambda x : [-(x[0]-1) + x[1], -(x[0]-1) - x[1]], 
	              lambda x : [-(x[0]+1) + x[1], -(x[0]+1) - x[1]]]

	# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
	data['beta'] = [lambda r,s : r * norm( expm(s*np.array([[-1,  1], [-1, -1]])) , np.inf), 
	 				lambda r,s : r * norm( expm(s*np.array([[-1, -1], [ 1, -1]])) , np.inf)]

	# Define upper and lower bounds on abstraction domain
	data['lb'] = [-2, -1]
	data['ub'] = [2, 1]

	# Define abstraction parameters
	data['eta'] = 0.1
	data['eps'] = 0.3
	data['tau'] = 1.1

	verify_bisim(data)
	verify_tree(data)

	ab = tree_abstr(data) 

	return ab

def simple_graph():
	g = nx.DiGraph()
	g.add_nodes_from([1,2,3,4,5,6,7,8])
	g.add_path([6,5,2,1,1], mode=1)
	g.add_path([8,7,2], mode=1)
	g.add_path([4,3,2], mode=1)
	g.add_path([1,2,3,6], mode=2)
	g.add_path([5,4,6] ,mode=2)
	g.add_path([6,7,8,8], mode=2)
	return g

def sum_modes(state, maxmode):
	return np.sum([state[ i * len(state)/maxmode : (i+1) * len(state)/maxmode] for i in range(maxmode)], axis=0)

def simulate(G, nodelist, init):

	# Construct subgraph we want to plot
	subgraph = G.subgraph(nodelist)
	subgraph_indices = [(G.nodes()).index(node) for node in subgraph.nodes()]

	# Initiate plot
	fig = plt.figure()
	ax = fig.gca()

	# Plot edges
	maxmode = max([d['mode'] for (u,v,d) in  G.edges(data=True)])

	# Get linear system description
	A,B = lin_syst(G)

	assert(maxmode * G.number_of_nodes() == A.shape[1])

	# feedback
	tree_roots = np.nonzero(A.diagonal())[0] # index of nonzero elements
	tree_roots_graph = tree_roots - np.array([i * len(G) for i in range(maxmode)])
	def u(state):
		u = np.zeros(len(state))
		u[tree_roots] = state[tree_roots]
		return u

	# Pre-compute plotting data
	tmax = 20
	xvec = np.zeros([len(init), tmax], dtype=float)
	xvec[:,0] = init
	for t in range(1,tmax):
		xvec[:,t] = A.dot(xvec[:,t-1]) + B.dot(u(xvec[:, t-1]))


	#################################
	######### PLOT THE GRAPH ########
	#################################
	pos = nx.spring_layout(subgraph)

	# Plot edges
	edgelists = [ [(u,v) for (u,v,d) in subgraph.edges(data=True) if d['mode'] == mode ]  for mode in range(1, maxmode+1)]
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(edgelists))))
	for i, edgelist in enumerate(edgelists):
		nx.draw_networkx_edges(subgraph, pos, ax=ax, edgelist = edgelist, edge_color = [next(colors)] * len(edgelist))

	# Plot initial set of nodes
	node_size = 300 * np.ones(len(subgraph_indices))
	for idx in tree_roots_graph:
		node_size[subgraph_indices.index(idx)] = 600

	nod = nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size = node_size )


	# Function that updates node colors
	def update(i):
		norm = mpl.colors.Normalize(vmin=0, vmax=np.max(xvec[:,i]))
		cmap = plt.cm.Blues
		m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		nod.set_facecolors(m.to_rgba(sum_modes(xvec[:,i], maxmode)[subgraph_indices] ))
		return nod

	ani = animation.FuncAnimation(fig, update, tmax, interval=1000, blit=False)
	ani.save('anim_output.mp4', fps=1)

	plt.show()

def main():
	# ab = example_abstr()
	# G = ab.graph

	G = simple_graph()

	init = np.zeros(len(G))
	for i in np.random.randint( len(G), size=50):
		init[i] += 1

	print "initial sum", sum(init)

	solve_lp(G, init)

	sg = G.subgraph(max(nx.strongly_connected_components(G), key=len))

	draw_modes(sg)
	plt.show()

if __name__ == '__main__':
	main()