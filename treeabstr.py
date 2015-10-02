import numpy as np

from numpy.linalg import norm
from scipy.linalg import expm
import scipy.integrate as integrate

import matplotlib.pyplot as plt

import itertools

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

class Cell(object):

	def __str__(self):
		return str(self.lb) + " <= x <= " + str(self.ub)

	def __init__(self, lb, ub):
		self.lb = lb;
		self.ub = ub;
		self.edgeOut1 = None
		self.edgeOut2 = None

	def mid(self):
		return (self.lb + self.ub) / 2

class Abstraction(object):
	def __init__(self, lb, ub, eta):
		self.lb = np.array(lb)
		self.ub = np.array(ub)
		self.n_dim = np.ceil((self.ub - self.lb)/eta).astype(np.uint64)
		self.cell_list = [None] * np.prod(self.n_dim)
		self.eta = eta

		for idx in range(len(self.cell_list)):
			midx = np.array(idx_to_midx(idx, self.n_dim))
			cell_lb = self.lb + midx * eta
			cell_ub = cell_lb + eta
			self.cell_list[idx] = Cell(cell_lb, cell_ub)

	def get_cell(self, midx):
		assert(np.all(midx < self.n_dim))
		return self.cell_list[midx_to_idx(midx, self.n_dim)]

	def get_midx_pt(self, pt):
		assert(len(pt) == len(self.n_dim))
		midx = np.floor((np.array(pt) - self.lb) / self.eta).astype(np.uint64)
		return midx

	def plot_2d(self):
		assert(len(self.lb) == 2)
		ax = plt.axes()
		for cell in self.cell_list:
			mid_this = cell.mid()
			mid_next = self.cell_list[midx_to_idx(cell.edgeOut1, self.n_dim)].mid()
			plt.plot(mid_this[0], mid_this[1], 'ro')
			ax.arrow(mid_this[0], mid_this[1],  mid_next[0] - mid_this[0], mid_next[1] - mid_this[1])

		plt.show()


def verify_bisim(data):
	assert(data['beta'](data['eps'], data['tau']) + data['eta']/2 <= data['eps'])

def tree_abstr(data):
	ab = Abstraction(data['lb'], data['ub'], data['eta'])

	dummy_vf = lambda z,t : data['vf'](z)

	for cell in ab.cell_list:
		x_fin = integrate.odeint(dummy_vf, cell.mid(), np.arange(0, data['tau'], data['tau']/100))
		cell.edgeOut1 = ab.get_midx_pt(x_fin[-1])

	return ab

def main():
	data = {}
	# Define a vector fields
	data['vf'] = lambda x : [-x[0] + x[1], -x[0] - x[1]]

	# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
	data['beta'] = lambda r,s : r * norm( expm(s*np.array([[-1, 1], [-1, -1]])) , np.inf)

	# Define upper and lower bounds on abstraction domain
	data['lb'] = [-1, -1]
	data['ub'] = [1, 1]

	# Define abstraction parameters
	data['eta'] = 0.1
	data['eps'] = 0.3
	data['tau'] = 0.5

	verify_bisim(data)
	ab = tree_abstr(data) 
	ab.plot_2d()

if __name__ == '__main__':
	main()