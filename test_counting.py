from nose.tools import *
import numpy as np
import scipy.sparse as sp
import networkx as nx
from itertools import product

from counting import CountingConstraint, ModeGraph, \
    MultiCountingProblem, _cycle_matrix
from optimization_wrappers import solve_mip


@raises(Exception)
def test_multimode_error():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3])

    G.add_edge(1, 3, modes=[1])
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


def test_mode_graph():
    G = ModeGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_path([3, 2, 1], modes=[0])
    G.add_path([1, 2, 3], modes=[1])

    np.testing.assert_equal(G.post(3, 0), 2)
    np.testing.assert_equal(G.post(2, 1), 3)


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
    cp = MultiCountingProblem(1)

    cp.graphs[0] = G
    cp.inits[0] = [0, 1, 6, 4, 7, 10, 2, 0]
    cp.T = 5

    cc1 = CountingConstraint(1)
    cc1.X[0] = set(product(G.nodes(), [0]))
    cc1.R = 16

    cc2 = CountingConstraint(1)
    cc2.X[0] = set(product(G.nodes(), [1]))
    cc2.R = 30 - 15

    cc3 = CountingConstraint(1)
    cc3.X[0] = set(product(G.nodes_with_selfloops(), [0, 1]))
    cc3.R = 0

    cp.constraints += [cc1, cc2, cc3]

    def outg(c):
        return [G[c[i]][c[(i + 1) % len(c)]]['modes'][0]
                for i in range(len(c))]

    cp.cycle_sets[0] = [zip(c, outg(c))
                        for c in nx.simple_cycles(nx.DiGraph(G))]

    cp.solve_prefix_suffix(solver='mosek')
    cp.test_solution()

    cp.solve_prefix_suffix(solver='gurobi')
    cp.test_solution()

    xi = sum([[i + 1] * cp.inits[0][i] for i in range(8)], [])
    for t in range(40):
        actions = cp.get_input([xi], t)
        for k1 in range(len(xi)):
            xi[k1] = G.post(xi[k1], actions[0][k1])


def test_multi():
    G1 = ModeGraph()
    G1.add_nodes_from([1, 2, 3])

    G1.add_path([1, 3, 3], modes=['on'])
    G1.add_path([1, 2], modes=['off'])
    G1.add_path([2, 2], modes=['on'])

    # Set up the mode-counting problem
    cp = MultiCountingProblem(2)
    cp.T = 3

    # Set up first class
    cp.graphs[0] = G1
    cp.inits[0] = [4, 0, 0]
    cp.cycle_sets[0] = [[(3, 'on')], [(2, 'on')]]

    # Set up second class
    cp.graphs[1] = G1
    cp.inits[1] = [4, 0, 0]
    cp.cycle_sets[1] = [[(3, 'on')], [(2, 'on')]]

    # Set up constraints

    # Count subsystems of class 0 that are at node `2` regardless of mode
    cc1 = CountingConstraint(2)
    cc1.X[0] = set([(2, 'on'), (2, 'off')])
    cc1.X[1] = set()
    cc1.R = 0

    cc2 = CountingConstraint(2)
    # Count subsystems of class 1 that are at node `3` regardless of mode
    cc2.X[0] = set()
    cc2.X[1] = set([(3, 'on'), (3, 'off')])
    cc2.R = 0

    cp.constraints += [cc1, cc2]

    cp.solve_prefix_suffix()

    cp.test_solution()

    xi = [[1, 1, 1, 1], [1, 1, 1, 1]]
    for t in range(7):
        actions = cp.get_input(xi, t)
        for k1 in range(4):
            xi[0][k1] = G1.post(xi[0][k1], actions[0][k1])

        for k2 in range(4):
            xi[1][k2] = G1.post(xi[1][k2], actions[1][k2])
        np.testing.assert_equal(xi, [[3, 3, 3, 3], [2, 2, 2, 2]])


def test_solvers():
    c = np.array([-1, -1])
    Aiq = sp.coo_matrix(np.array([[0.5, 1],
                                  [1, 0]]))
    biq = np.array([1.75, 1.5])
    Aeq = sp.coo_matrix((0, 2))
    beq = np.array([])

    sol = solve_mip(c, Aiq, biq, Aeq, beq, [0, 1], solver='mosek')
    np.testing.assert_equal(sol['status'], 2)
    np.testing.assert_equal(sol['x'], [1, 1])

    sol = solve_mip(c, Aiq, biq, Aeq, beq, [0, 1], solver='gurobi')
    np.testing.assert_equal(sol['status'], 2)
    np.testing.assert_equal(sol['x'], [1, 1])

    sol = solve_mip(c, Aiq, biq, Aeq, beq, [], solver='mosek')
    np.testing.assert_equal(sol['status'], 2)
    np.testing.assert_almost_equal(sol['x'], [1.5, 1])

    sol = solve_mip(c, Aiq, biq, Aeq, beq, [], solver='gurobi')
    np.testing.assert_equal(sol['status'], 2)
    np.testing.assert_almost_equal(sol['x'], [1.5, 1])


