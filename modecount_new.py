import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import deque

from optimization_wrappers import *


class ModeGraph(nx.DiGraph):

    def __init__(self):
        super(ModeGraph, self).__init__()
        self.order_fcn = lambda v: self.nodes().index(v)

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

        return sp.bmat([[mode_matrix(m) for m in range(self.M())]])

    def index_matrix(self, C):
        """Compute matrix Psi_C s.t.
           Psi_C alpha
           is a G-vector"""
        row_idx = [self.order_fcn(v) for v, _ in C]
        col_idx = range(len(C))
        vals = np.ones(len(C))
        return scipy.sparse.coo_matrix(
            (vals, (row_idx, col_idx)), shape=(self.K(), len(C))
        )

    def modes(self):
        return set().union(*[d['modes']
                           for _, _, d in self.edges_iter(data=True)])

    def M(self):
        return max(self.modes()) + 1

    def K(self):
        return len(self)

    def check_valid(self):
        for v in self.nodes_iter():
            M = [self[v][w]['modes'] for w in self[v]]
            if sum(len(m) for m in M) != len(set().union(*M)):
                raise Exception("Nondeterministic graph")

        modes = set().union(*[d['modes']
                              for _, _, d in self.edges_iter(data=True)])
        if modes != set(range(max(modes) + 1)):
            raise Exception("Modes incorrectly numbered")


class SingleCountingProblem(object):
    """Class representing a graph counting problem"""
    def __init__(self):

        # Problem data
        self.G = None
        self.constraints = []
        self.init = None
        self.cycle_set = None
        self.T = None

        # Solution data
        self.x = None
        self.u = None
        self.assignments = None

    def solve_prefix_suffix(self):
        """Solve counting problem given an initial state, a horizon T,
        and a set of (augmented) cycles"""

        if self.T is None:
            raise Exception("No problem horizon `T` specified")

        if self.init is None:
            raise Exception("No initial condition `init` specified")

        if self.cycle_set is None:
            raise Exception("No cycle set `cycle_set` specified")

        if self.G is None:
            raise Exception("ModeGraph `G` not set")

        try:
            self.G.check_valid()
        except Exception as e:
            raise e

        # Todo:
        #   - prune graph to find invariant set

        K = self.G.K()
        M = self.G.M()
        L = len(self.constraints)
        J = len(self.cycle_set)
        cycle_set = self.cycle_set
        T = self.T

        # variables: u[0], ..., u[T-1], x[1], ..., x[T], a[0], ..., a[C-1]
        N_u = T * K * M   # input vars
        N_x = T * K   # state vars
        N_a = sum(len(C) for C in cycle_set)  # cycle vars
        N_b = L * J    # bound vars

        N_tot = N_u + N_x + N_a + N_b

        A_eq1_u, A_eq1_x, b_eq1 = \
            generate_prefix_dyn_cstr(self.G, self.T, self.init)
        A_eq2_x, A_eq2_a, b_eq2 = \
            generate_prefix_suffix_cstr(self.G, self.T, self.cycle_set)
        A_eq1_b = sp.coo_matrix((A_eq1_x.shape[0], N_b))

        A_eq = sp.bmat([[A_eq1_u, A_eq1_x, None, A_eq1_b],
                        [None, A_eq2_x, A_eq2_a, None]])
        b_eq = np.hstack([b_eq1, b_eq2])

        A_iq_list = []
        b_iq_list = []

        for l in range(len(self.constraints)):
            X, R = self.constraints[l]

            # Get T constraints
            A_iq1_u_j, b_iq1 = \
                generate_prefix_counting_cstr(self.G, self.T, X, R)
            A_iq1_x_j = sp.coo_matrix((T, N_x))

            # Get N_a + 1 constraints
            A_iq2_a_j_1, A_iq2_b_j_1, b_iq2_j_1, \
                A_iq2_a_j_2, A_iq2_b_j_2, b_iq2_j_2 = \
                generate_suffix_counting_cstr(self.cycle_set, X, R)

            A_iq2_a_j = sp.bmat([[A_iq2_a_j_1],
                                 [A_iq2_a_j_2]])

            # Need padding to select correct slack variable
            b_head = sp.coo_matrix((N_a, l * J))
            b_tail = sp.coo_matrix((N_a, (L - 1 - l) * J))
            A_iq2_b_j = sp.bmat([[b_head, A_iq2_b_j_1, b_tail],
                                 [None, A_iq2_b_j_2, None]])

            b_iq2 = np.hstack([b_iq2_j_1, b_iq2_j_2])

            # Stack it!
            A_iq_list.append(
                sp.bmat([[A_iq1_u_j, A_iq1_x_j, None, None],
                         [None, None, A_iq2_a_j, A_iq2_b_j]])
            )
            b_iq_list.append(
                np.hstack([b_iq1, b_iq2])
            )

        A_iq = sp.bmat([[A] for A in A_iq_list])
        b_iq = np.hstack(b_iq_list)

        # Solve it!
        sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq)

        # Extract solution (if valid)
        if sol['status'] == 2:
            self.u = np.array(sol['x'][0:N_u]).reshape(T, K * M).transpose()
            self.x = np.hstack([
                np.array(self.init).reshape(len(self.init), 1),
                np.array(sol['x'][N_u:N_u + N_x]).reshape(T, K).transpose()
            ])

            self.cycle_set = cycle_set

            cycle_lengths = [len(C) for C in cycle_set]
            self.assignments = [sol['x'][N_u + N_x + an: N_u + N_x + an + dn]
                                for an, dn in zip(np.cumsum([0] +
                                                  cycle_lengths[:-1]),
                                                  cycle_lengths)
                                ]

        return sol['status']

    def test_solution(self):
        """Test if a solution satisfies the constraints"""

        # Check dynamics
        np.testing.assert_almost_equal(
            self.x[:, 1:],
            self.G.system_matrix().dot(self.u)
        )

        # Check control constraints
        np.testing.assert_almost_equal(
            self.x[:, :-1],
            _id_stacked(self.G.K(), self.G.M()).dot(self.u)
        )

        # Check prefix counting bounds
        for X, R in self.constraints:
            for u in self.u.transpose():
                assert(sum(u[self.G.order_fcn(v) + m * self.G.K()]
                           for (v, m) in X) <= R
                       )

        # Check prefix-suffix connection
        assgn_sum = np.zeros(self.G.K())
        for C, a in zip(self.cycle_set, self.assignments):
            for (Ci, ai) in zip(C, a):
                assgn_sum[self.G.order_fcn(Ci[0])] += ai

        np.testing.assert_almost_equal(
            self.x[:, -1],
            assgn_sum
        )

        # Check suffix counting bounds
        ass_rot = [deque(a) for a in self.assignments]
        for t in range(1000):  # enough to go up to LCM(cycle lengths)
            for X, R in self.constraints:
                assert(sum(np.inner(_cycle_row(C, X), a)
                           for C, a in zip(self.cycle_set, ass_rot)
                           ) <= R)
            for a in ass_rot:
                a.rotate()


class MultiCountingProblem(object):
    """For multiple classes"""
    def __init__(self):
        self.graphs = []
        self.constraints = []
        self.inits = []
        self.cycle_sets = []
        self.T = None

        self.u = []
        self.x = []
        self.assignments = []

    def solve_prefix_suffix(self):
        if self.T is None:
            raise Exception("No problem horizon `T` specified")

        if len(self.inits) == 0:
            raise Exception("No initial condition `inits` specified")

        if len(self.cycle_sets) == 0:
            raise Exception("No cycle set `cycle_sets` specified")

        if len(self.graphs) == 0:
            raise Exception("ModeGraphs `graphs` not set")

        for G in self.graphs:
            try:
                G.check_valid()
            except Exception as e:
                raise e

        for constraint in self.constraints:
            X, R = constraint
            if len(X) != len(self.graphs):
                raise Exception("Each constraint needs to be on \
                                 form (X_1, ..., X_P) for classes p")
            for Xg, G in zip(X, self.graphs):
                if len(Xg) == 0:
                    continue
                Xg_x, Xg_u = zip(*Xg)
                if not all([x in G.nodes() for x in Xg_x]):
                    print set(Xg_x) <= set(G.nodes())
                    raise Exception("State-part of constraint invalid")
                if not all([m in G.modes() for m in Xg_u]):
                    raise Exception("Mode-part of constraint invalid")

        if (len(self.inits) != len(self.graphs)) or \
           (len(self.cycle_sets) != len(self.graphs)):
            raise Exception("Need same number of graphs, \
                           initial conditions, and cycle sets")

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

        # Add dynamics constraints
        A_eq_list = []
        b_eq_list = []

        for g in range(len(self.graphs)):
            G = self.graphs[g]
            init = self.inits[g]
            cycle_set = self.cycle_sets[g]

            A_eq1_u, A_eq1_x, b_eq1 = \
                generate_prefix_dyn_cstr(G, T, init)
            A_eq2_x, A_eq2_a, b_eq2 = \
                generate_prefix_suffix_cstr(G, T, cycle_set)
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
            X, R = self.constraints[l]

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
                    generate_prefix_counting_cstr(self.graphs[g], T, X[g], R)
                A_iq1_list.append(
                    sp.bmat([[A_iq1_u, sp.coo_matrix((T, N_x_list[g] + N_a_list[g] + N_b_list[g]))]])
                )

                # Suffix counting
                A_iq2_a, A_iq2_b, b_iq2, A_iq3_a, A_iq3_b, b_iq3 = \
                    generate_suffix_counting_cstr(self.cycle_sets[g], X[g], R)

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
                    sp.bmat([[sp.coo_matrix((1, N_u_list[g] + N_x_list[g])), A_iq3_a, A_iq3_b]])
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
        for X, R in self.constraints:
            for t in range(1000):  # enough to do T + LCM(cycle_lengths)
                assert(self.mode_count(X, t) <= R)

    def mode_count(self, X, t):
        X_count = 0
        for g in range(len(self.graphs)):
            G = self.graphs[g]
            if t < self.T:
                u_t_g = self.u[g][:, t]
                X_count += sum(u_t_g[G.order_fcn(v) + m * G.K()]
                               for (v, m) in X[g])
            else:
                ass_rot = [deque(a) for a in self.assignments[g]]
                for a in ass_rot:
                    a.rotate(t - self.T)
                X_count += sum(np.inner(_cycle_row(C, X[g]), a)
                               for C, a in zip(self.cycle_sets[g],
                                               ass_rot))
        return X_count

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
               if m not in G.node_modes(v)]
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

    assert A_eq_u.shape[0] == len(b_eq)
    assert A_eq_x.shape[0] == len(b_eq)

    return A_eq_u, A_eq_x, b_eq


def generate_prefix_counting_cstr(G, T, X, R):
    """Generate inequalities (47a) for prefix counting constraints"""
    K = G.K()
    M = G.M()

    # variables: u[0], ..., u[T-1]

    col_idx = [G.order_fcn(v) + m * K for (v, m) in X]
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

    A_eq_x = scipy.sparse.bmat(
        [[sp.coo_matrix((K, K * (T - 1))), -sp.identity(K)]]
    )

    A_eq_a = scipy.sparse.bmat([Psi_mats])
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
