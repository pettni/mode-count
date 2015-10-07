import unittest

import numpy as np
import networkx as nx
import scipy.linalg

from treeabstr import midx_to_idx, idx_to_midx, Abstraction, lin_syst, cycle_rows, cycle_matrix, sum_modes, cycle_indices, \
					  simple_graph, solve_lp

class treeabstrTests(unittest.TestCase):

	def test_midx_to_idx(self):
		for i in range(5):
			for j in range(4):
				self.assertEquals(midx_to_idx((i,j), (5,4)), 4*i+j)

		for i in range(5):
			for j in range(4):
				for k in range(3):
					self.assertEquals(midx_to_idx((i,j,k), (5,4,3)), 3*(4*i+j) + k)
					self.assertEquals(idx_to_midx(midx_to_idx((i,j,k), (5,4,3)), (5,4,3)), (i,j,k))

	def test_abstraction(self):
		ab = Abstraction([-1, -1], [1, 1], 1)
		self.assertEquals(len(ab.graph), 9)

		c1 = ab.get_midx_pt((0.2, 0.2))
		self.assertEquals(ab.graph.node[c1]['lb'], (-0.5,-0.5))
		self.assertEquals(ab.graph.node[c1]['ub'], (0.5,0.5))

		c2 = ab.get_midx_pt((0.2, -0.7))
		self.assertEquals(ab.graph.node[c2]['lb'], (-0.5,-1.5))
		self.assertEquals(ab.graph.node[c2]['ub'], (0.5,-0.5))

	def test_lin_syst(self):
		g = nx.DiGraph()
		g.add_nodes_from([1,2,3])
		g.add_path([2,1], mode=1)
		g.add_path([3,1], mode=1)
		g.add_path([1,2,3], mode=2)
		A,B = lin_syst(g)

		T = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
		Tbar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
		self.assertTrue(np.all(A.toarray() == scipy.linalg.block_diag(T, Tbar)))
		self.assertTrue(np.all(B.toarray() == np.vstack([np.hstack([-T, T]), np.hstack([Tbar, -Tbar])])  ))

	def test_cycle_rows_matrix(self):
		g = nx.DiGraph()
		g.add_nodes_from([1,2,3])
		g.add_path([2,1], mode=1)
		g.add_path([3,1], mode=1)
		g.add_path([1,2,3], mode=2)
		cr = cycle_rows(g, [1,2,3], 2)
		self.assertTrue(np.all(next(cr) ==  [1, 1, 0]))
		self.assertTrue(np.all(next(cr) ==  [1, 0, 1]))
		self.assertTrue(np.all(next(cr) ==  [0, 1, 1]))

		cm = cycle_matrix(g, [1,2,3], 2)
		self.assertTrue(np.all(cm == [[1, 1, 0], [1, 0, 1], [0, 1, 1]]))

	def test_sum_modes(self):
		state = [1,1,1,0,0,0]
		a = sum_modes(state, 2)
		self.assertTrue(np.all(a == [1,1,1]))

		state = [1,1,1,0,2,0]
		a = sum_modes(state, 2)
		self.assertTrue(np.all(a == [1,3,1]))

		state = [1,1,1,0,2,0,4,5,6]
		a = sum_modes(state, 3)
		self.assertTrue(np.all(a == [5,8,7]))

	def test_cycle_indices(self):
		g = nx.DiGraph()
		g.add_nodes_from([1,2,3,4,5,6])
		g.add_path([2,3,4,5,2])

		ci = cycle_indices(g, [2,3,4,5]).todense()
		self.assertTrue(np.all(ci == np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0,0,0,0]]) ))

	def test_solve_lp(self):
		G = simple_graph()

		T = 5
		Kdes = 13.
		mode = 1
		init = [ 5,  9,  4,  3,  8,  7,  6,  8]
		states, u_finite, cycles, alphas = solve_lp(G, init, T, Kdes, mode)

		self.assertEquals(len(cycles), 2)
		self.assertEquals(set(cycles[0]), set([2,6,3,7]))
		self.assertEquals(set(cycles[1]), set([3,6,5,4]))
		


if __name__ == '__main__':
	unittest.main()