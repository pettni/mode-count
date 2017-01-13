import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import deque

from optimization_wrappers import *


class ModeGraph(nx.DiGraph):

    def __init__(self):
        super(ModeGraph, self).__init__()
        self.order_fcn = lambda v: self.nodes().index(v)
        self.mode_list = None

    def set_order_fcn(self, fcn):
        """Specify an order function `fcn` : G.nodes() --> range(K)"""
        if set(range(self.K())) != set(fcn(v) for v in self.nodes_iter()):
            raise Exception("Invalid order function")
        self.order_fcn = fcn

    def node_modes(self, v):
        """All outgoing modes at `v`"""
        return sum([self[v][w]['modes'] for w in self[v]], [])

    def system_matrix(self):
        """Matrix B s.t. x(t+1) = Bx(t)"""
        def mode_matrix(m):
            data = np.array([[1, self.order_fcn(v), self.order_fcn(u)]
                            for u, v, data in self.edges_iter(data=True)
                            if m in data['modes']])
            return sp.coo_matrix((data[:, 0],
                                 (data[:, 1], data[:, 2])),
                                 shape=(self.K(), self.K()))

        return sp.bmat([[mode_matrix(m) for m in self.modes()]])

    def index_matrix(self, C):
        """Compute matrix Psi_C s.t.
           Psi_C alpha
           is a vector w.r.t. graph indexing"""
        row_idx = [self.order_fcn(v) for v, _ in C]
        col_idx = range(len(C))
        vals = np.ones(len(C))
        return sp.coo_matrix(
            (vals, (row_idx, col_idx)), shape=(self.K(), len(C))
        )

    def modes(self):
        """Return set of all modes"""
        if self.mode_list is None:
            self.mode_list = list(set().union(*[d['modes']
                                              for _, _, d
                                              in self.edges_iter(data=True)]))
        return self.mode_list

    def mode(self, m):
        """Return mode with index `m`"""
        return self.modes()[m]

    def index_of_mode(self, mode):
        return self.modes().index(mode)

    def M(self):
        """Return total number of modes"""
        return len(self.modes())

    def K(self):
        """Return total number of nodes"""
        return len(self)

    def check_valid(self):
        """Verify that graph is deterministic"""
        for v in self.nodes_iter():
            M = [self[v][w]['modes'] for w in self[v]]
            if sum(len(m) for m in M) != len(set().union(*M)):
                raise Exception("Nondeterministic graph")

    def post(self, node, mode):
        """Return next state for node `node` and action `mode`"""
        if mode not in self.node_modes(node):
            raise Exception("Invalid action " + str(mode) +
                            " at " + str(node))
        for w in self[node]:
            if mode in self[node][w]['modes']:
                return w


class CountingConstraint(object):
    def __init__(self, order):
        self.X = [set()] * order
        self.order = order
        self.R = 0


class MultiCountingProblem(object):
    """For multiple classes"""
    def __init__(self, order):
        self.order = order
        self.graphs = [None] * order
        self.constraints = []
        self.inits = [None] * order
        self.cycle_sets = [None] * order
        self.T = None

        self.u = []
        self.x = []
        self.assignments = []


    def check_well_defined(self):
        # Check input data
        if self.T is None:
            raise Exception("No problem horizon `T` specified")

        if None in self.inits:
            raise Exception("Initial conditions `inits` missing")

        if None in self.cycle_sets:
            raise Exception("Cycle sets `cycle_sets` missing")

        if None in self.graphs:
            raise Exception("Graphs `graphs` missing")

        # Check that graphs are deterministic
        for G in self.graphs:
            try:
                G.check_valid()
            except Exception as e:
                raise e

        # Check validity of constraints
        for cc in self.constraints:
            if cc.order != self.order:
                raise Exception("CountingConstraints must be of order " +
                                str(self.order))
            for g in range(len(self.graphs)):
                if len(cc.X[g]) == 0:
                    continue
                Xg_x, Xg_u = zip(*cc.X[g])
                if not all([x in G.nodes() for x in Xg_x]):
                    print set(Xg_x) <= set(G.nodes())
                    raise Exception("State-part of constraint invalid")
                if not all([m in G.modes() for m in Xg_u]):
                    raise Exception("Mode-part of constraint invalid")

        # Check that cycles are valid
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            for cycle in self.cycle_sets[g]:
                for j in range(len(cycle)):
                    v1 = cycle[j][0]
                    v2 = cycle[(j + 1) % len(cycle)][0]
                    m = cycle[j][1]
                    if not G.has_edge(v1, v2) or m not in G[v1][v2]['modes']:
                        raise Exception("Found invalid cycle")

    def solve_prefix_suffix(self):

        self.check_well_defined()

        # Variables for each g in self.graphs:
        #  v_g := u_g[0] ... u_g[T-1] x_g[0] ... x_g[T-1]
        #         a_g[0] ... a_g[C1-1] b[0] ... b[LJ-1]
        # These are stacked horizontally as
        #  v_0 v_1 ... v_G-1

        L = len(self.constraints)
        T = self.T
        J_list = [len(cycle_set) for cycle_set in self.cycle_sets]

        # Variable counts for each class g
        N_u_list = [T * G.K() * G.M() for G in self.graphs]   # input vars
        N_x_list = [T * G.K() for G in self.graphs]   # state vars
        N_a_list = [sum(len(C) for C in cs) for cs in self.cycle_sets]
        N_b_list = [L * J for J in J_list]    # bound vars

        N_tot = sum(N_u_list) + sum(N_x_list) + sum(N_a_list) + sum(N_b_list)

        # Add dynamics constraints, should be block diagonalized
        A_eq_list = []
        b_eq_list = []

        for g in range(len(self.graphs)):
            A_eq1_u, A_eq1_x, b_eq1 = \
                generate_prefix_dyn_cstr(self.graphs[g], T, self.inits[g])
            A_eq2_x, A_eq2_a, b_eq2 = \
                generate_prefix_suffix_cstr(self.graphs[g], T,
                                            self.cycle_sets[g])
            A_eq1_b = sp.coo_matrix((A_eq1_u.shape[0], N_b_list[g]))

            A_eq_list.append(
                sp.bmat([[A_eq1_u, A_eq1_x, None, A_eq1_b],
                         [None, A_eq2_x, A_eq2_a, None]])
            )
            b_eq_list.append(
                np.hstack([b_eq1, b_eq2])
            )

        A_eq = sp.block_diag(A_eq_list)
        b_eq = np.hstack(b_eq_list)

        # Add counting constraints
        A_iq_list = []
        b_iq_list = []
        for l in range(len(self.constraints)):
            cc = self.constraints[l]

            # Count over classes: Should be stacked horizontally
            A_iq1_list = []

            # Bounded by slack vars: Should be block diagonalized
            A_iq2_list = []
            b_iq2_list = []

            # Count over bound vars for each class: Should be stacked
            # horizontally
            A_iq3_list = []

            for g in range(len(self.graphs)):
                # Prefix counting
                A_iq1_u, b_iq1 = \
                    generate_prefix_counting_cstr(self.graphs[g], T,
                                                  cc.X[g], cc.R)
                A_iq1_list.append(
                    sp.bmat([[A_iq1_u, sp.coo_matrix((T, N_x_list[g] +
                                                      N_a_list[g] +
                                                      N_b_list[g]))]])
                )

                # Suffix counting
                A_iq2_a, A_iq2_b, b_iq2, A_iq3_a, A_iq3_b, b_iq3 = \
                    generate_suffix_counting_cstr(self.cycle_sets[g],
                                                  cc.X[g], cc.R)

                b_head2 = sp.coo_matrix((N_a_list[g], l * J_list[g]))
                b_tail2 = sp.coo_matrix((N_a_list[g], (L - 1 - l) * J_list[g]))
                A_iq2_b = sp.bmat([[b_head2, A_iq2_b, b_tail2]])
                b_head3 = sp.coo_matrix((1, l * J_list[g]))
                b_tail3 = sp.coo_matrix((1, (L - 1 - l) * J_list[g]))
                A_iq3_b = sp.bmat([[b_head3, A_iq3_b, b_tail3]])

                A_iq2_u = sp.coo_matrix((N_a_list[g], N_u_list[g]))
                A_iq2_x = sp.coo_matrix((N_a_list[g], N_x_list[g]))

                A_iq2_list.append(
                    sp.bmat([[A_iq2_u, A_iq2_x, A_iq2_a, A_iq2_b]])
                )
                b_iq2_list.append(b_iq2)

                A_iq3_list.append(
                    sp.bmat([[sp.coo_matrix((1, N_u_list[g] + N_x_list[g])),
                              A_iq3_a, A_iq3_b]])
                )

            # Stack horizontally
            A_iq_list.append(sp.bmat([A_iq1_list]))
            b_iq_list.append(b_iq1)

            # Stack by block
            A_iq_list.append(sp.block_diag(A_iq2_list))
            b_iq_list.append(np.hstack(b_iq2_list))

            # Stack horizontally
            A_iq_list.append(sp.bmat([A_iq3_list]))
            b_iq_list.append(b_iq3)

        # Stack everything vertically
        if len(A_iq_list) > 0:
            A_iq = sp.bmat([[A] for A in A_iq_list])
            b_iq = np.hstack(b_iq_list)
        else:
            A_iq = sp.coo_matrix((0, N_tot))
            b_iq = np.zeros(0)

        # Solve it
        sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq)
        # Extract solution (if valid)
        if sol['status'] == 2:
            idx0 = 0
            for g in range(len(self.graphs)):
                self.u.append(
                    np.array(sol['x'][idx0:idx0 + N_u_list[g]])
                      .reshape(T, self.graphs[g].K() * self.graphs[g].M())
                      .transpose()
                )
                self.x.append(
                    np.hstack([
                        np.array(self.inits[g]).reshape(len(self.inits[g]), 1),
                        np.array(sol['x'][idx0 + N_u_list[g]:
                                          idx0 + N_u_list[g] + N_x_list[g]])
                          .reshape(T, self.graphs[g].K()).transpose()
                    ])
                )

                cycle_lengths = [len(C) for C in self.cycle_sets[g]]
                self.assignments.append(
                    [sol['x'][idx0 + N_u_list[g] + N_x_list[g] + an:
                              idx0 + N_u_list[g] + N_x_list[g] + an + dn]
                     for an, dn in zip(np.cumsum([0] +
                                       cycle_lengths[:-1]),
                                       cycle_lengths)]
                )
                idx0 += N_u_list[g] + N_x_list[g] + N_a_list[g] + N_b_list[g]
        return sol['status']

    def test_solution(self):
        for g in range(len(self.graphs)):
            # Check dynamics
            np.testing.assert_almost_equal(
                self.x[g][:, 1:],
                self.graphs[g].system_matrix().dot(self.u[g])
            )

            # Check control constraints
            np.testing.assert_almost_equal(
                self.x[g][:, :-1],
                _id_stacked(self.graphs[g].K(), self.graphs[g].M())
                    .dot(self.u[g])
            )

            # Check prefix-suffix connection
            assgn_sum_g = np.zeros(self.graphs[g].K())
            for C, a in zip(self.cycle_sets[g], self.assignments[g]):
                for (Ci, ai) in zip(C, a):
                    assgn_sum_g[self.graphs[g].order_fcn(Ci[0])] += ai

            np.testing.assert_almost_equal(
                self.x[g][:, -1],
                assgn_sum_g
            )

        # Check counting constraints
        for cc in self.constraints:
            for t in range(1000):  # enough to do T + LCM(cycle_lengths)
                assert(self.mode_count(cc.X, t) <= cc.R)

    def mode_count(self, X, t):
        """When a solution has been found, return the `X`-count
        at time `t`"""
        X_count = 0
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            if t < self.T:
                u_t_g = self.u[g][:, t]
                X_count += sum(u_t_g[G.order_fcn(v) +
                                     G.index_of_mode(m) * G.K()]
                               for (v, m) in X[g])
            else:
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)
                X_count += sum(np.inner(_cycle_row(C, X[g]), a)
                               for C, a in zip(self.cycle_sets[g],
                                               ass_rot))
        return X_count

    def get_input(self, xi_list, t):
        """When an integer solution has been found, given individual states
        `xi_list` at time t, return individual inputs `sigma`"""
        if self.u is None:
            raise Exception("No solution available")

        actions = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]
            N_g = np.sum(self.inits[g])

            actions_g = [None] * N_g

            if t < self.T:
                # We are in prefix; states are available
                x_g = self.x[g][:, t]
                u_g = self.u[g][:, t]

            else:
                # We are in suffix, must build x_g, u_g from cycle assignments
                u_g = np.zeros(G.K() * G.M())
                x_g = np.zeros(len(self.graphs[g]))

                # Rotate assignments
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)

                for assgn, c in zip(ass_rot,
                                    self.cycle_sets[g]):
                    x_g += self.graphs[g].index_matrix(c) \
                               .dot(assgn).flatten()
                    for ai, ci in zip(assgn, c):
                        u_g[G.order_fcn(ci[0]) +
                            G.index_of_mode(ci[1]) * G.K()] += ai

            # Assert that xi_list agrees with aggregate solution
            xi_sum = np.zeros(len(self.graphs[g]))
            for xi in xi_list[g]:
                xi_sum[self.graphs[g].order_fcn(xi)] += 1
            try:
                np.testing.assert_almost_equal(xi_sum,
                                               x_g)
            except:
                raise Exception("States don't agree with aggregate" +
                                " at time " + str(t))

            for n in range(N_g):
                k = G.order_fcn(xi_list[g][n])
                u_state = [u_g[k + G.K() * m] for m in range(G.M())]
                m = next(i for i in range(len(u_state))
                         if u_state[i] >= 1)
                actions_g[n] = G.mode(m)
                u_g[k + G.K() * m] -= 1
            actions.append(actions_g)

        return actions


####################################
#  Constraint-generating functions #
####################################


def generate_prefix_dyn_cstr(G, T, init):
    """Generate equalities (47c), (47e) for prefix dynamics"""
    K = G.K()
    M = G.M()

    # variables: u[0], ..., u[T-1], x[1], ..., x[T]

    # Obtain system matrix
    B = G.system_matrix()

    # (47c)
    # T*K equalities
    A_eq1_u = sp.block_diag((B,) * T)
    A_eq1_x = sp.block_diag((sp.identity(K),) * T)
    b_eq1 = np.zeros(T * K)

    # (47e)
    # T*K equalities
    A_eq2_u = sp.block_diag((_id_stacked(K, M),) * T)
    A_eq2_x = sp.bmat([[sp.coo_matrix((K, K * (T - 1))),
                        sp.coo_matrix((K, K))],
                       [sp.block_diag((sp.identity(K),) * (T - 1)),
                        sp.coo_matrix((K * (T - 1), K))]
                       ])
    b_eq2 = np.hstack([init, np.zeros((T - 1) * K)])

    # Forbid non-existent modes
    # T * len(ban_idx) equalities
    ban_idx = [G.order_fcn(v) + m * K
               for v in G.nodes_iter()
               for m in range(M)
               if G.mode(m) not in G.node_modes(v)]
    A_eq3_u_part = sp.coo_matrix(
        (np.ones(len(ban_idx)), (range(len(ban_idx)), ban_idx)),
        shape=(len(ban_idx), K * M)
    )
    A_eq3_u = sp.block_diag((A_eq3_u_part,) * T)
    A_eq3_x = sp.coo_matrix((T * len(ban_idx), T * K))
    b_eq3 = np.zeros(T * len(ban_idx))

    # Stack everything
    A_eq_u = sp.bmat([[A_eq1_u],
                     [A_eq2_u],
                     [A_eq3_u]])
    A_eq_x = sp.bmat([[-A_eq1_x],
                      [-A_eq2_x],
                      [A_eq3_x]])
    b_eq = np.hstack([b_eq1, b_eq2, b_eq3])

    return A_eq_u, A_eq_x, b_eq


def generate_prefix_counting_cstr(G, T, X, R):
    """Generate inequalities (47a) for prefix counting constraints"""
    K = G.K()
    M = G.M()

    # variables: u[0], ..., u[T-1]

    col_idx = [G.order_fcn(v) + G.index_of_mode(m) * K for (v, m) in X]
    if len(col_idx) == 0:
        # No matches
        return sp.coo_matrix((T, T * K * M)), R * np.ones(T)

    val = np.ones(len(col_idx))
    row_idx = np.zeros(len(col_idx))
    A_pref_cc = sp.coo_matrix(
        (val, (row_idx, col_idx)), shape=(1, K * M)
    )

    A_iq_u = sp.block_diag((A_pref_cc,) * T)
    b_iq = R * np.ones(T)

    return A_iq_u, b_iq


def generate_prefix_suffix_cstr(G, T, cycle_set):
    """Generate K equalities (47d) that connect prefix and suffix"""
    K = G.K()

    # Variables x[1] ... x[T] a[0] ... a[C-1]

    Psi_mats = [G.index_matrix(C) for C in cycle_set]

    A_eq_x = sp.bmat(
        [[sp.coo_matrix((K, K * (T - 1))), -sp.identity(K)]]
    )

    A_eq_a = sp.bmat([Psi_mats])
    b_eq = np.zeros(K)

    return A_eq_x, A_eq_a, b_eq


def generate_suffix_counting_cstr(cycle_set, X, R):
    """Generate inequalities (47b) for suffix counting"""

    # Variables a[0] ... a[C-1] + slack vars b[c][l]

    J = len(cycle_set)
    N_cycle_tot = sum(len(C) for C in cycle_set)

    # First set: A_iq1_a a + A_iq1_b b \leq b_iq1
    # guarantee that count in each cycle is less than
    # slack var
    A_iq1_a = sp.block_diag(tuple([_cycle_matrix(C, X)
                            for C in cycle_set]))

    A_iq1_b = sp.block_diag(tuple([-np.ones([len(C), 1])
                            for C in cycle_set]))
    b_iq1 = np.zeros(N_cycle_tot)

    # Second set: A_iq2_b b \leq b_iq2
    # guarantees that sum of slack vars
    # less than R
    A_iq2_a = sp.coo_matrix((1, N_cycle_tot))
    A_iq2_b = sp.coo_matrix((np.ones(J), (np.zeros(J), range(J))),
                            shape=(1, J))

    b_iq2 = np.array([R])

    return A_iq1_a, A_iq1_b, b_iq1, A_iq2_a, A_iq2_b, b_iq2

####################
# Helper functions #
####################


def _id_stacked(K, M):
    """Return the (K x MK) sparse matrix [I I ... I]"""
    return sp.coo_matrix((np.ones(K * M),
                         ([k
                           for k in range(K)
                           for m in range(M)],
                          [k + m * K
                           for k in range(K)
                           for m in range(M)]
                          )
                          ),
                         shape=(K, K * M))


def _cycle_row(C, X):
    """Compute vector v s.t. <v,alpha> is
       the number of subsystems in `X`"""
    return [1 if C[i] in X else 0 for i in range(len(C))]


def _cycle_matrix(C, X):
    """Compute matrix A_C s.t. A_C alpha is
       all rotated numbers of subsystems in `X`"""
    idx = deque([i for i in range(len(C)) if C[i] in X])
    vals = np.ones(len(idx) * len(C))
    row_idx = []
    col_idx = []
    for i in range(len(C)):
        row_idx += (i,) * len(idx)
        col_idx += [(j - i) % len(C) for j in idx]
    return sp.coo_matrix(
        (vals, (row_idx, col_idx)), shape=(len(C), len(C))
    )
