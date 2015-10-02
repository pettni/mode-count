import unittest

import numpy as np

from treeabstr import midx_to_idx, idx_to_midx, Abstraction

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
		self.assertEquals(len(ab.cell_list), 4)

		c1 = ab.get_cell(ab.get_midx_pt((0.5, 0.5)))
		self.assertEquals(tuple(c1.lb), (0,0))
		self.assertEquals(tuple(c1.ub), (1,1))

		c2 = ab.get_cell(ab.get_midx_pt((0.5, -0.3)))
		self.assertEquals(tuple(c2.lb), (0,-1))
		self.assertEquals(tuple(c2.ub), (1,0))

if __name__ == '__main__':
	unittest.main()