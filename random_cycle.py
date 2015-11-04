import networkx as nx
import numpy as np

def mychoice(list):
	return list[np.random.randint(len(list))]

def diff(a, b):
	""" 
		Return set difference of lists a,b as list
	"""
	b = set(b)
	return [aa for aa in a if aa not in b]

def dfs(G, node, visited, forbidden, min_length, mode_weight, temp_forbidden = set([])):
	""" 
		Do a depth first search
			G 				- graph
			node 			- node to start at
			visited 		- list of previously visited nodes
			forbidden 		- nodes that can't be in cycle
			min_length 		- minimal cycle length
			mode_weight 	- probab. to continue in same mode
			temp_forbidden  - used to temporarily exclude cycles that are too short
	"""

	allowed_nodes = diff(G.successors(node), forbidden.union(temp_forbidden))

	if len(allowed_nodes) == 0:
		# Reached leaf
		return False

	# choose random successor
	if len(visited) >= 2 and len(allowed_nodes) >= 2:
		current_mode = G[visited[-2]][visited[-1]]['mode']
		next_modes = [G[node][next_node]['mode'] for next_node in allowed_nodes ]
		if current_mode in next_modes:
			# 
			ind = next_modes.index(current_mode)
			next_node_same = allowed_nodes[ind]
			if np.random.random(1) < mode_weight:
				# stick with same mode
				next_node = next_node_same
			else:
				# choose other mode uniformly
				next_node = mychoice(diff(allowed_nodes, set([next_node_same])))
		else:
			# can not continue with current mode
			next_node = mychoice(allowed_nodes)	
	else:
		next_node = mychoice(allowed_nodes)

	if next_node in visited:
		# found a cycle
		cycle = visited[ visited.index(next_node): ]

		if len(cycle) > min_length:
			# rotate it
			rot_ind = np.argmin(cycle)
			return cycle[rot_ind:] + cycle[:rot_ind]
		else:
			# must choose other node
			return dfs(G, node, visited, forbidden, min_length, mode_weight, set([next_node]))

	# no cycle, continue down graph
	test = dfs(G, next_node, visited + [next_node], forbidden, min_length, mode_weight)
	if test:
		# found a cycle downstream
		return test 
	else:
		# need to pick another vertex!
		return dfs(G, node, visited, forbidden.union(set([next_node])), min_length, mode_weight)

def random_cycle(G, min_length = 2, mode_weight = 0.5):
	'''
	Generate a random simple cycle in the graph G

	Inputs:
	  G 	      : graph to search in
	  		         class: networkx DiGraph
	  min_length  : minimal cycle length to consider
	  				 type: int
	  mode_weight : when randomly selecting successors, select same mode with 
	  				this probability
	  				 type: double in interval [0,1]
	Returns:
	  cycle  	  : list of nodes in G
	  False		  : if no simple cycle longer than min_length exists

	Comments: - exhaustive search can be slow, best for graphs with an abundance of cycles
			  - algorithm is a randomized DFS 
			  - the probability for a given cycle to be selected is unknown and is not uniform
	'''
	cycle = False
	tried_nodes = set([])
	while not cycle: 
		remaining_nodes = diff(G.nodes(), tried_nodes)
		if len(remaining_nodes) == 0:
			# graph doesnt have cycles
			return False
		init_node = mychoice(remaining_nodes)
		cycle = dfs(G, init_node, [init_node], set([]), min_length, mode_weight)
		tried_nodes = tried_nodes.union([init_node])
	verify_cycle(G, cycle)
	return cycle

def verify_cycle(G, cyc):
	for i in range(len(cyc)):
		this_node = cyc[i]
		next_node = cyc[(i+1) % len(cyc)]
		assert(next_node in G.successors(this_node))