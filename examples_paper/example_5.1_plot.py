import cPickle as pickle

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

unsafe_x = 0.2
unsafe_y = 0.2
margin = 0.15

G, sol_data = pickle.load(open('example_5.1.save', 'rb') )

numcycles = len(sol_data['cycles'])

H = max(nx.strongly_connected_component_subgraphs(G), key=len)

pos = {}
for node, attr in H.nodes_iter(data=True):
	pos[node] = attr['mid']

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

currentAxis.add_patch(Rectangle( (-unsafe_x, -unsafe_y), 2*unsafe_x, 2*unsafe_y, color='red', alpha=0.3) )
currentAxis.add_patch(Rectangle( (-(unsafe_x+margin), -(unsafe_y+margin)), 2*(unsafe_x+margin), 2*(unsafe_y+margin), color='red', alpha=0.3) )

#Options
# plt.axis('off')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig('example_5.1_fig.pdf', format='pdf')
