import networkx as nx
import numpy as np
from random import choice

def diff(a, b):
	""" 
		Return set difference of lists a,b as list
	"""
	b = set(b)
	return [aa for aa in a if aa not in b]

def dfs(G, node, visited, forbidden):
	""" 
		Do a depth first search
	"""
	allowed_nodes = diff(G.successors(node),forbidden)

	if len(allowed_nodes) == 0:
		# Reached leaf
		return False

	# choose random successor
	next_node = choice(allowed_nodes)

	if next_node in visited:
		# found a cycle
		cycle = visited[ visited.index(next_node): ]

		# rotate it
		rot_ind = np.argmin(cycle)
		return cycle[rot_ind:] + cycle[:rot_ind]

	# continue looking
	test = dfs(G, next_node, visited + [next_node], forbidden )
	if test:
		# found a cycle downstream
		return test 
	else:
		# need to pick another vertex!
		return dfs(G, node, visited, forbidden.union(set([next_node])) )

def random_cycle(G, ml = 2):
	cycle = False
	tried_nodes = set([])
	while not cycle: 
		remaining_nodes = diff(G.nodes(), tried_nodes)
		if len(remaining_nodes) == 0:
			# graph doesnt have cycles
			return False

		init_node = choice(remaining_nodes)
		cycle = dfs(G, init_node, [init_node], set([]))
		tried_nodes = tried_nodes.union([init_node])
	return cycle

def main():
	G = nx.DiGraph()
	G.add_nodes_from([1,2,3,4,5,6,7,8,9,10])
	G.add_path([1,2,3,1])
	G.add_path([1,2,3,4,5,6,7,1])
	G.add_path([7,8,9])
	G.add_path([7,10,9])

	print random_cycle(G)


if __name__ == '__main__':
	main()