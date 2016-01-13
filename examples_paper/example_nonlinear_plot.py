import cPickle as pickle

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

unsafe_radius = 0.15
margin = 0.1

G, sol_data = pickle.load(open('example_nonlinear.save', 'rb') )

H = max(nx.strongly_connected_component_subgraphs(G), key=len)

numcycles = len(sol_data['cycles'])

pos = {}
for node, attr in H.nodes_iter(data=True):
	pos[node] = scalefac*attr['mid']

def edges(cycle):
	return [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]

nx.draw_networkx_nodes(H, pos, nodelist = list(set.union(*[set(cyc) for cyc in sol_data['cycles'][:numcycles] ])), node_color = 'black', alpha=0.5)

colors = ['red', 'green', 'blue', 'yellow', 'purple', 'magenta', 'cyan']
for cyclenum in range(numcycles):
	nx.draw_networkx_edges(H, pos, edgelist = edges(sol_data['cycles'][cyclenum]) , edge_color = colors[cyclenum % len(colors)], style='solid', width=1.5)

xmin = min([val[0] for val in pos.itervalues()]) - 0.1
xmax = max([val[0] for val in pos.itervalues()]) + 0.1
ymin = min([val[1] for val in pos.itervalues()]) - 0.1
ymax = max([val[1] for val in pos.itervalues()]) + 0.1

currentAxis = plt.gca()

currentAxis.add_patch(Circle( (0, 0), radius=unsafe_radius, color='red', alpha=0.3) )
currentAxis.add_patch(Circle( (0, 0), radius=(unsafe_radius + margin ), color='red', alpha=0.3) )

#Options
plt.axis('off')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig('example_nonlinear_fig.pdf', format='pdf')