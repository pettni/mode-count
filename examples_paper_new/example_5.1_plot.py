import cPickle as pickle

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

G, sol_data = pickle.load(open('example_5.1.save', 'rb') )

H = max(nx.strongly_connected_component_subgraphs(G), key=len)

edgelist1 = [(u,v) for (u,v,d) in H.edges(data=True) if d['mode'] == 1 ]
edgelist2 = [(u,v) for (u,v,d) in H.edges(data=True) if d['mode'] == 2 ]

# pos = nx.spring_layout(H)

scalefac = 10

pos = {}
for node, attr in H.nodes_iter(data=True):
	pos[node] = scalefac*attr['mid']

# nx.draw_networkx_edges(H, pos, edgelist = edgelist1, edge_color = 'red', alpha=0.2)
# nx.draw_networkx_edges(H, pos, edgelist = edgelist2, edge_color = 'blue', alpha=0.2)

def edges(cycle):
	return [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]

nx.draw_networkx_nodes(H, pos, nodelist = list(set.union(*[set(cyc) for cyc in sol_data['cycles'] ])), node_color = 'black', alpha=0.5)

nx.draw_networkx_edges(H, pos, edgelist = edges(sol_data['cycles'][0]) , edge_color = 'red', style='solid', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(sol_data['cycles'][1]) , edge_color = 'green', style='solid', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(sol_data['cycles'][2]) , edge_color = 'blue', style='solid', width=1.5)
nx.draw_networkx_edges(H, pos, edgelist = edges(sol_data['cycles'][3]) , edge_color = 'yellow', style='solid', width=1.5)
# nx.draw_networkx_edges(H, pos, edgelist = edges(sol_data['cycles'][4]) , edge_color = 'purple', style='solid', width=1.5)

xmin = min([val[0] for val in pos.itervalues()]) - 0.1 * scalefac
xmax = max([val[0] for val in pos.itervalues()]) + 0.1 * scalefac
ymin = min([val[1] for val in pos.itervalues()]) - 0.1 * scalefac
ymax = max([val[1] for val in pos.itervalues()]) + 0.1 * scalefac

currentAxis = plt.gca()

currentAxis.add_patch(Rectangle( (-scalefac*0.15, -scalefac*0.15), scalefac*0.3, scalefac*0.3, color='red', alpha=0.3) )
currentAxis.add_patch(Rectangle( (-scalefac*0.3, -scalefac*0.3), scalefac*0.6, scalefac*0.6, color='red', alpha=0.3) )

#Options
plt.axis('off')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig('ex2d_graph.pdf', format='pdf')