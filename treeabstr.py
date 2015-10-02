import numpy as np

from numpy.linalg import norm
from scipy.linalg import expm
import scipy.integrate as integrate
import random

import networkx as nx

import matplotlib.pyplot as plt

import itertools

def draw_modes(G):
	maxmode = max([d['mode'] for (u,v,d) in  G.edges(data=True)]) + 1
	edgelists = [ [(u,v) for (u,v,d) in G.edges(data=True) if d['mode'] == mode ]  for mode in range(maxmode)]
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(edgelists) + 1)))

	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_color = [next(colors)] * G.number_of_nodes() )
	for i, edgelist in enumerate(edgelists):
		nx.draw_networkx_edges(G, pos, edgelist = edgelist, edge_color = [next(colors)] * len(edgelist))


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
		self.lb -= (self.lb + eta/2) % eta  # make sure that zero is in middle of cell
		self.ub = np.array(ub, dtype=np.float64)
		self.ub += (self.ub - self.lb) % eta # make domain number of eta's
		self.nextMode = 0

		# 
		self.n_dim = np.round((self.ub - self.lb)/eta).astype(np.uint64)

		# Create a graph and populate it with nodes
		self.graph = nx.DiGraph()
		for midx in itertools.product(*[range(dim) for dim in self.n_dim]):
			cell_lb = self.lb + np.array(midx) * eta
			cell_ub = cell_lb + eta
			self.graph.add_node(midx, lb = cell_lb, ub = cell_ub, mid = (cell_lb + cell_ub) / 2)

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
		dummy_vf = lambda z,t : vf(z)
		for node, attr in self.graph.nodes_iter(data=True):
			x_fin = integrate.odeint(dummy_vf, attr['mid'], np.arange(0, tau, tau/100))
			if self.contains(x_fin[-1]):
				midx = self.get_midx_pt(x_fin[-1])
				self.graph.add_edge( node, midx, mode = self.nextMode )
			else:
				print "Warning: there are transitions out of domain"
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

def main():
	ab = example_abstr()
	# plot_treeineq(data)
	# ab.plot_planar()	

if __name__ == '__main__':
	main()