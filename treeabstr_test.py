import unittest

import numpy as np
import networkx as nx
import scipy.linalg

from treeabstr import midx_to_idx, idx_to_midx, Abstraction, lin_syst

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

		T = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
		Tbar = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
		self.assertTrue(np.all(A.toarray() == scipy.linalg.block_diag(T, Tbar)))
		self.assertTrue(np.all(B.toarray() == np.vstack([np.hstack([-T, T]), np.hstack([Tbar, -Tbar])])  ))

if __name__ == '__main__':
	unittest.main()