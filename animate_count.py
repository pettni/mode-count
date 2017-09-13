import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from itertools import product

def animate_count(cp, tmax, subgraph_nodes = None):
	G = cp.graphs[0]
	B = G.system_matrix()

	xvec = np.zeros([G.K(), tmax+1], dtype=float)
	uvec = np.zeros([G.K()*G.M(), tmax+1], dtype=float)

	xvec[:,0] = cp.inits[0]
	for t in range(tmax):
		uvec[:, t] = cp.get_aggregate_input(t)[0]
		xvec[:,t+1] = B.dot(uvec[:, t]);

	#######################################
	#######################################

	if not subgraph_nodes:
		subgraph_nodes = [v for v in G.nodes_iter() if np.any(xvec[G.order_fcn(v), tmax-3:tmax]) > 0]

	subG = G.subgraph(sum([G.predecessors(v) for v in subgraph_nodes], []) )

	ANIM_STEP = 30

	# Initiate plot
	fig, ax = plt.subplots()
	plt.axis('off')

	pos = nx.spring_layout(subG, k=2./np.sqrt(len(subG)))

	# Time text
	time_template = 'time = %d'
	time_text = plt.text(0.05, 0.05, '', transform=ax.transAxes)

	# Graph
	# node_handle = nx.draw_networkx_nodes(subG, pos, nodelist=subgraph_nodes, node_color='black', node_size=100)
	edges_handle = nx.draw_networkx_edges(subG, pos, arrows=False)

	# Mode text
	mode_template = 'count %d: %.1f'
	counting_list = []
	for i in range(len(cp.constraints)):
		counting_list.append(ax.text(0.02 + i*0.25, 0.9, '', transform=ax.transAxes))
	plt.xlim([-0.7, 0.7])
	plt.ylim([-0.7, 0.7])

	# Edge animation
	edge_anim = []
	edge_list = []
	for v, m in product(subG.nodes_iter(), G.modes()):
		w = G.post(v, m)
		if w in subgraph_nodes:
			edge_anim.append(*ax.plot(pos[v], pos[w], marker='o', color = 'w', markersize=10))
			edge_list.append((v,m,w))

	def animate(i):
		x_ind = i/ANIM_STEP
		anim_ind = i % ANIM_STEP

		x_i = xvec[:,x_ind]
		u_i = uvec[:,x_ind]
		t = float(anim_ind)/ANIM_STEP

		# Update node colors
		norm = mpl.colors.Normalize(vmin=-3, vmax=np.max(x_i))
		cmap = plt.cm.Blues
		mapping = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		# node_handle.set_facecolor(mapping.to_rgba(x_i))
		
		# Print time
		time_text.set_text(time_template % x_ind)

		# Counts
		for i, ct in enumerate(counting_list):
			ct.set_text(mode_template % (i , cp.mode_count(cp.constraints[i].X, x_ind)))

		# Edge animation
		for i in range(len(edge_anim)):
			v,m,w = edge_list[i]
			num_edge = u_i[G.K() * G.index_of_mode(m) + G.order_fcn(v)]
			if num_edge > 1e-5:
				edge_anim[i].set_visible(True)
				edge_anim[i].set_color(mapping.to_rgba(num_edge))
				edge_anim[i].set_data(t * pos[w][0] + (1-t) * pos[v][0], t * pos[w][1] + (1-t) * pos[v][1])
			else:
				edge_anim[i].set_visible(False)


		return [time_text] + counting_list + edge_anim

	ani = animation.FuncAnimation(fig, animate, np.arange(1, tmax * ANIM_STEP), blit=True)
	plt.show()
	# writer = animation.writers['ffmpeg'](fps=ANIM_STEP)
	# ani.save('test.mp4',writer=writer,dpi=100)

