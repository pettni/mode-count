import itertools
import numpy as np
import scipy
import matplotlib.pyplot as plt

from counting import ModeGraph


class Abstraction(object):
    """Discrete abstraction of the hyper box defined by *lower_bounds*
       and *upper_bounds*. Time discretization is *tau* and space
       discretization is given by *eta*. Mode transitions are added
       with :py:func:`add_mode`."""
    def __init__(self, lower_bounds, upper_bounds, eta, tau):
        self.eta = eta
        self.tau = float(tau)

        self.lower_bounds = np.array(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.array(upper_bounds, dtype=np.float64)
        self.upper_bounds += (self.upper_bounds - self.lower_bounds) \
            % eta  # make domain number of eta's
        self.nextMode = 1

        # number of discrete points along each dimension
        self.n_dim = np.round((self.upper_bounds - self.lower_bounds) / eta) \
            .astype(int)

        # Create a graph and populate it with nodes
        self.graph = ModeGraph()
        for midx in itertools.product(*[range(dim) for dim in self.n_dim]):
            cell_lower_bounds = self.lower_bounds + np.array(midx) * eta
            cell_upper_bounds = cell_lower_bounds + eta
            self.graph.add_node(midx,
                                lower_bounds=tuple(cell_lower_bounds),
                                upper_bounds=tuple(cell_upper_bounds),
                                mid=(cell_lower_bounds +
                                     cell_upper_bounds) / 2)
        self.graph.set_order_fcn(self.node_to_idx)

    def __len__(self):
        ''' Size of abstraction '''
        return len(self.graph)

    def point_to_midx(self, point):
        ''' Return the node multiindex corresponding to the continuous
            point *point*. '''
        assert(len(point) == len(self.n_dim))
        if not self.contains(point):
            raise('Point outside domain')
        midx = np.floor((np.array(point) - self.lower_bounds) /
                        self.eta).astype(np.uint64)
        return tuple(midx)

    def contains(self, point):
        ''' Return ``True`` if *point* is within the abstraction domain,
        ``False`` otherwise. '''
        if np.any(self.lower_bounds >= point) or \
           np.any(self.upper_bounds <= point):
            return False
        return True

    def plot_planar(self):
        ''' Plot a 2D abstraction. '''
        assert(len(self.lower_bounds) == 2)
        ax = plt.axes()
        for node, attr in self.graph.nodes_iter(data=True):
            plt.plot(attr['mid'][0], attr['mid'][1], 'ro')
        for n1, n2, mode in self.graph.edges_iter(data='mode'):
            mid1 = self.graph.node[n1]['mid']
            mid2 = self.graph.node[n2]['mid']
            col = 'b' if mode == 1 else 'g'
            ax.arrow(mid1[0], mid1[1],
                     mid2[0] - mid1[0],
                     mid2[1] - mid1[1],
                     fc=col, ec=col)

        plt.show()

    def add_mode(self, vf, mode_name=None):
        ''' Add new dynamic mode to the abstraction, given by
        the vector field *vf*. '''

        if mode_name is None:
            mode_name = self.nextMode
            self.nextMode += 1

        def dummy_vf(z, t):
            return vf(z)
        tr_out = 0
        for node, attr in self.graph.nodes_iter(data=True):
            x_fin = scipy.integrate.odeint(dummy_vf,
                                           attr['mid'],
                                           np.arange(0, self.tau,
                                                     self.tau / 100))
            if self.contains(x_fin[-1]):
                midx = self.point_to_midx(x_fin[-1])
                if self.graph.has_edge(node, midx):
                    # Edge already present, append mode
                    self.graph[node][midx]['modes'] += mode_name
                else:
                    # Create new edge with single mode
                    self.graph.add_edge(node, midx, modes=[mode_name])
            else:
                tr_out += 1
        if tr_out > 0:
            print "Warning: ", tr_out, " transitions out of ", \
                  len(self.graph), " in mode ", mode_name, \
                  " go out of domain"

    def node_to_idx(self, node):
        ''' Given a node at discrete multiindex :math:`(x,y,z)`,
            return the index :math:`L_z ( L_y x + y ) + z`,
            where :math:`L_z, L_y` are the (discrete) lengths of the
            hyper box domain,
            and correspondingly for higher/lower dimensions. The function
            is a 1-1 mapping between
            the nodes in the abstraction and the positive integers,
            and thus suitable as
            order_function in :py:func:`prefix_suffix_feasible`. '''
        assert len(node) == len(self.n_dim)
        ret = np.int64(node[0])
        for i in range(1, len(self.n_dim)):
            ret *= np.int64(self.n_dim[i])
            ret += np.int64(node[i])
        return ret

    def idx_to_node(self, idx):
        ''' Inverse of :py:func:`node_to_idx` '''
        assert(idx < np.product(self.n_dim))
        node = [0] * len(self.n_dim)
        for i in reversed(range(len(self.n_dim))):
            node[i] = int(idx % self.n_dim[i])
            idx = np.floor(idx / self.n_dim[i])
        return tuple(node)


def verify_bisim(beta, tau, eta, eps, delta, K):
    ''' Given a :math:`\mathcal{KL}`-function *beta* for a continuous
    system, return ``True`` if the abstraction
    created with time discretization *tau* and space discretization *eta*
    passes the *eps*-approximate bisimilarity
    test, and ``False`` otherwise. '''
    return (beta(eps, tau) + eta / 2 +
            (delta / K) * (np.exp(K * tau) - 1) <= eps)
