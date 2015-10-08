import unittest

import numpy as np
import networkx as nx
import scipy.linalg

from modecount import Abstraction, lin_syst, _cycle_rows, _cycle_matrix, _sum_modes, _cycle_indices, \
					  synthesize

class modecountTests(unittest.TestCase):

	def test_abstraction(self):
		ab = Abstraction([-1, -1], [1, 1], 1, 1, 1)
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

	def test__cycle_rows_matrix(self):
		g = nx.DiGraph()
		g.add_nodes_from([1,2,3])
		g.add_path([2,1], mode=1)
		g.add_path([3,1], mode=1)
		g.add_path([1,2,3], mode=2)
		cr = _cycle_rows(g, [1,2,3], 2)
		self.assertTrue(np.all(next(cr) ==  [1, 1, 0]))
		self.assertTrue(np.all(next(cr) ==  [1, 0, 1]))
		self.assertTrue(np.all(next(cr) ==  [0, 1, 1]))

		cm = _cycle_matrix(g, [1,2,3], 2)
		self.assertTrue(np.all(cm == [[1, 1, 0], [1, 0, 1], [0, 1, 1]]))

	def test__sum_modes(self):
		state = [1,1,1,0,0,0]
		a = _sum_modes(state, 2)
		self.assertTrue(np.all(a == [1,1,1]))

		state = [1,1,1,0,2,0]
		a = _sum_modes(state, 2)
		self.assertTrue(np.all(a == [1,3,1]))

		state = [1,1,1,0,2,0,4,5,6]
		a = _sum_modes(state, 3)
		self.assertTrue(np.all(a == [5,8,7]))

	def test__cycle_indices(self):
		g = nx.DiGraph()
		g.add_nodes_from([1,2,3,4,5,6])
		g.add_path([2,3,4,5,2])

		ci = _cycle_indices(g, [2,3,4,5]).todense()
		self.assertTrue(np.all(ci == np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0,0,0,0]]) ))

	def test_solve_lp(self):
		G = nx.DiGraph()
		G.add_nodes_from([1,2,3,4,5,6,7,8])

		G.add_path([6,5,2,1,1], mode=2)
		G.add_path([8,7,4], mode=2)
		G.add_path([4,3,2], mode=2)
		G.add_path([1,2,3,6], mode=1)
		G.add_path([5,4,6] ,mode=1)
		G.add_path([6,7,8,8], mode=1)

		# Plot it
		# draw_modes(G)
		# plt.show()

		# Specify initial system distribution (sums to 30)

		T = 5
		Kdes = 16.
		mode = 1
		init = [0, 1, 6, 4, 7, 10, 2, 0]
		sol = synthesize(G, init, T, Kdes, mode)

		cycles = sorted(sol['cycles'], key=len)

		self.assertEquals(len(cycles), 2)
		self.assertEquals(set(cycles[0]), set([4,6,5]))
		self.assertEquals(set(cycles[1]), set([3,6,5,4]))

if __name__ == '__main__':
	unittest.main()