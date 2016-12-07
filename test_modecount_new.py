from nose.tools import *
import networkx as nx
import numpy as np
from itertools import product

from modecount_new import SingleCountingProblem, ModeGraph, _cycle_matrix


@raises(Exception)
def test_multimode_error():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3])

    G.add_edge(1, 3, modes=[1])
    G.add_edge(1, 2, modes=[1])
    G.check_valid()


@raises(Exception)
def test_mudenumber_error():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3])

    G.add_edge(1, 3, modes=[2])
    G.add_edge(1, 2, modes=[1])
    G.check_valid()


@raises(Exception)
def test_orderfcn_error():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3])

    G.add_edge(1, 3, modes=[1])
    G.add_edge(1, 2, modes=[0])
    G.set_order_fcn(lambda n: n)


def test_system_matrix():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3])

    G.add_path([3, 2, 1], modes=[0])
    G.add_path([1, 2, 3], modes=[1])

    A = G.system_matrix().todense()

    np.testing.assert_equal(
        A,
        np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    )


def test_cycle_indices():
    G = ModeGraph()
    G.add_nodes_from([5, 1, 2, 3, 4])

    def order_fcn(n):
        if n == 5:
            return 0
        else:
            return n

    G.add_edge(2, 1, modes=[0])
    G.add_edge(1, 1, modes=[0])
    G.add_edge(1, 2, modes=[1])
    G.set_order_fcn(order_fcn)

    cycle1 = [(2, 0), (1, 0), (1, 1)]

    A1 = G.index_matrix(cycle1).todense()

    np.testing.assert_equal(
        A1,
        np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
    )


def test_cycle_matrix():
    cycle1 = [(2, 0), (1, 0), (1, 1)]

    A1 = _cycle_matrix(cycle1, [(1, 0), (2, 0)]).todense()

    np.testing.assert_equal(
        A1,
        np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]])
    )

    cycle2 = [(1, 1), (2, 0)]
    A2 = _cycle_matrix(cycle2, [(1, 0), (2, 0)]).todense()

    np.testing.assert_equal(
        A2,
        np.array([[0, 1],
                  [1, 0]])
    )


def test_comprehensive():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])

    G.add_path([6, 5, 2, 1, 1], modes=[1])
    G.add_path([8, 7, 4], modes=[1])
    G.add_path([4, 3, 2], modes=[1])
    G.add_path([1, 2, 3, 6], modes=[0])
    G.add_path([5, 4, 6], modes=[0])
    G.add_path([6, 7, 8, 8], modes=[0])

    # Set up the mode-counting problem
    cp = SingleCountingProblem()

    cp.G = G
    cp.init = [0, 1, 6, 4, 7, 10, 2, 0]
    cp.T = 5

    cp.constraints.append((list(product(G.nodes(), [0])), 16))
    cp.constraints.append((list(product(G.nodes(), [1])), 30 - 15))
    cp.constraints.append((list(product(G.nodes_with_selfloops(), [0, 1])), 0))

    def outg(c):
        return [G[c[i]][c[(i + 1) % len(c)]]['modes'][0]
                for i in range(len(c))]

    cp.cycle_set = [zip(c, outg(c)) for c in nx.simple_cycles(nx.DiGraph(G))]

    cp.solve_prefix_suffix()

    cp.test_solution()
