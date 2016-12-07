import numpy as np
import scipy.sparse as sp
from collections import deque

from modecount import _sparse_vstack
from optimization_wrappers import *


class SingleCountingProblem(object):
    """Class representing a graph counting problem"""
    def __init__(self, G):
        for v in G.nodes_iter():
            M = [G[v][w]['modes'] for w in G[v]]
            if sum(len(m) for m in M) != len(set().union(*M)):
                raise Exception("Nondeterministic graph")

        modes = set().union(*[d['modes']
                              for _, _, d in G.edges_iter(data=True)])
        if modes != set(range(max(modes) + 1)):
            raise Exception("Modes incorrectly numbered")

        # Problem data
        self.G = G
        self.K = len(G)
        self.M = 1 + max(modes)
        self.constraints = []

        # Settings
        self.order_fcn = lambda v: self.G.nodes().index(v)

        # Saved solution
        self.x = None
        self.u = None
        self.cycle_set = None
        self.assignments = None

    def node_modes(self, v):
        return sum([self.G[v][w]['modes'] for w in self.G[v]], [])

    def add_constraint(self, X, R):
        """Add a counting constraint"""
        self.constraints.append((X, R))

    def set_order_fcn(self, fcn):
        """Specify an order function `fcn` : G.nodes() --> range(K)"""
        if set(range(self.K)) != set(fcn(v) for v in self.G.nodes_iter()):
            raise Exception("Invalid order function")
        self.order_fcn = fcn

    def system_matrix(self):
        """Matrix B s.t. x(t+1) = Bx(t)"""
        def mode_matrix(m):
            data = np.array([[1, self.order_fcn(v), self.order_fcn(u)]
                            for u, v, data in self.G.edges_iter(data=True)
                            if m in data['modes']])
            return sp.coo_matrix((data[:, 0],
                                 (data[:, 1], data[:, 2])),
                                 shape=(self.K, self.K))

        return sp.bmat([[mode_matrix(m) for m in range(self.M)]])

    def _id_stacked(self):
        """Return the (K x MK) sparse matrix [I I ... I]"""
        return sp.coo_matrix((np.ones(self.K * self.M),
                             ([k
                               for k in range(self.K)
                               for m in range(self.M)],
                              [k + m * self.K
                               for k in range(self.K)
                               for m in range(self.M)]
                              )
                              ),
                             shape=(self.K, self.K * self.M))

    def cycle_indices(self, C):
        """Compute matrix Psi_C s.t.
           Psi_C alpha
           is a G-vector"""
        row_idx = [self.order_fcn(v) for v, _ in C]
        col_idx = range(len(C))
        vals = np.ones(len(C))
        return scipy.sparse.coo_matrix(
            (vals, (row_idx, col_idx)), shape=(self.K, len(C))
        )

    def cycle_row(self, C, X):
        """Compute vector v s.t. <v,alpha> is
           the number of subsystems in X"""
        return [1 if C[i] in X else 0 for i in range(len(C))]

    def cycle_matrix(self, C, X):
        """Compute matrix A_C s.t. A_C alpha is
           all rotated numbers of subsystems in X"""
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

    def generate_prefix_dyn_cstr(self, init, T):
        """Generate (in)equalities for prefix dynamics"""
        K = len(self.G)
        M = self.M

        # variables: u[0], ..., u[T-1], x[1], ..., x[T]
        N_u = T * K * M   # input vars
        N_x = T * K   # state vars

        # Obtain system matrix
        B = self.system_matrix()

        # (47c)
        # T*K equalities
        A_eq1_u = sp.block_diag((B,) * T)
        A_eq1_x = sp.block_diag((sp.identity(K),) * T)
        b_eq1 = np.zeros(T * K)

        # (47e)
        # T*K equalities
        A_eq2_u = sp.block_diag((self._id_stacked(),) * T)
        A_eq2_x = sp.bmat([[sp.coo_matrix((K, K * (T - 1))),
                            sp.coo_matrix((K, K))],
                           [sp.block_diag((sp.identity(K),) * (T - 1)),
                            sp.coo_matrix((K * (T - 1), K))]
                           ])
        b_eq2 = np.hstack([init, np.zeros((T - 1) * K)])

        # Forbid non-existent modes
        # len(ban_idx) equalities
        ban_idx = [self.order_fcn(v) + m * K
                   for v in self.G.nodes_iter()
                   for m in range(M)
                   if m not in self.node_modes(v)]
        A_eq3_u = sp.coo_matrix(
            (np.ones(len(ban_idx)), (range(len(ban_idx)), ban_idx)),
            shape=(len(ban_idx), N_u)
        )
        A_eq3_x = sp.coo_matrix((len(ban_idx), N_x))
        b_eq3 = np.zeros(T * len(ban_idx))

        A_eq_u = sp.bmat([[A_eq1_u],
                         [A_eq2_u],
                         [A_eq3_u]])
        A_eq_x = sp.bmat([[-A_eq1_x],
                          [-A_eq2_x],
                          [A_eq3_x]])
        b_eq = np.hstack([b_eq1, b_eq2, b_eq3])

        return A_eq_u, A_eq_x, b_eq

    def generate_prefix_counting_cstr(self, T):
        """Generate T * L inequalities for prefix counting constraints"""
        K = len(self.G)
        M = self.M

        # variables: u[0], ..., u[T-1]
        N_u = T * K * M   # input vars

        A_iq_u = sp.coo_matrix((0, N_u))
        b_iq = np.array([])

        for X, R in self.constraints:
            # T inequalities for each l
            col_idx = [self.order_fcn(v) + m * K
                       for v in self.G.nodes_iter()
                       for m in range(M)
                       if (v, m) in X]

            val = np.ones(len(col_idx))
            row_idx = np.zeros(len(col_idx))
            A_pref_cc = sp.coo_matrix(
                (val, (row_idx, col_idx)), shape=(1, K * M)
            )
            A_iq_u = sp.bmat([[A_iq_u],
                             [sp.block_diag((A_pref_cc,) * T)]])
            b_iq = np.hstack([b_iq, R * np.ones(T)])

        return A_iq_u, b_iq

    def generate_prefix_suffix_cstr(self, T, cycle_set):
        """Generate K equalities that connect prefix and suffix"""
        K = self.K

        # Variables x[1] ... x[T] a[0] ... a[C-1]

        Psi_mats = [self.cycle_indices(C) for C in cycle_set]

        A_eq_x = scipy.sparse.bmat(
            [[sp.coo_matrix((K, K * (T - 1))), -sp.identity(K)]]
        )

        A_eq_a = scipy.sparse.bmat([Psi_mats])
        b_eq = np.zeros(K)

        return A_eq_x, A_eq_a, b_eq

    def generate_suffix_counting_cstr(self, T, cycle_set):
        """Generate inequalities for suffix counting"""

        L = len(self.constraints)
        J = len(cycle_set)

        # Variables a[0] ... a[C-1] + slack vars

        N_cycle_tot = sum([len(C) for C in cycle_set])
        N_bound = L * J    # bound vars

        A_iq_a = sp.coo_matrix((0, N_cycle_tot))
        A_iq_b = sp.coo_matrix((0, N_bound))
        b_iq = np.array([])

        for l in range(L):
            # (N_cycle_tot + 1) inequalities for each l
            X = self.constraints[l][0]
            R = self.constraints[l][1]

            A_cyc1_a = sp.block_diag(tuple([self.cycle_matrix(C, X)
                                     for C in cycle_set]))

            A_cyc1_b1 = sp.coo_matrix((N_cycle_tot, l * J))

            A_cyc1_b2 = sp.block_diag(tuple([-np.ones([len(C), 1])
                                      for C in cycle_set]))
            A_cyc1_b3 = sp.coo_matrix((N_cycle_tot, (L - 1 - l) * J))

            A_cyc1_b = sp.bmat([[A_cyc1_b1, A_cyc1_b2, A_cyc1_b3]])

            A_cyc2_b1 = sp.coo_matrix((1, l * J))
            A_cyc2_b2 = sp.coo_matrix((np.ones(J), (np.zeros(J), range(J))),
                                      shape=(1, J))
            A_cyc2_b3 = sp.coo_matrix((1, (L - 1 - l) * J))

            A_cyc2_b = sp.bmat([[A_cyc2_b1, A_cyc2_b2, A_cyc2_b3]])

            A_iq_a = sp.bmat([[A_iq_a],
                              [A_cyc1_a],
                              [sp.coo_matrix((1, N_cycle_tot))]])

            A_iq_b = sp.bmat([[A_iq_b],
                              [A_cyc1_b],
                              [A_cyc2_b]])
            b_iq = np.hstack([b_iq, np.zeros(N_cycle_tot), R])

        return A_iq_a, A_iq_b, b_iq

    def solve_prefix_suffix(self, init, T, cycle_set):
        """Solve counting problem given an initial state, a horizon T,
        and a set of (augmented) cycles"""

        # Todo:
        #   - prune graph to find invariant set

        K = len(self.G)
        M = self.M
        L = len(self.constraints)
        J = len(cycle_set)

        # variables: u[0], ..., u[T-1], x[1], ..., x[T], a[0], ..., a[C-1]
        N_u = T * K * M   # input vars
        N_x = T * K   # state vars
        N_cycle_tot = sum([len(C) for C in cycle_set])  # cycle vars
        N_bound = L * J    # bound vars

        N_tot = N_u + N_x + N_cycle_tot + N_bound

        A_eq1_u, A_eq1_x, b_eq1 = \
            self.generate_prefix_dyn_cstr(init, T)
        A_iq1_u, b_iq1 = self.generate_prefix_counting_cstr(T)
        A_eq2_x, A_eq2_a, b_eq2 = \
            self.generate_prefix_suffix_cstr(T, cycle_set)
        A_iq2_a, A_iq2_b, b_iq2 = \
            self.generate_suffix_counting_cstr(T, cycle_set)

        # Filling 0 matrices
        A_iq1_x = sp.coo_matrix((A_iq1_u.shape[0], N_x))
        A_eq2_b = sp.coo_matrix((A_eq2_x.shape[0], N_bound))

        # Stack everything
        A_iq = sp.bmat([[A_iq1_u,  A_iq1_x,  None,    None],
                        [None,     None,     A_iq2_a, A_iq2_b]])
        b_iq = np.hstack([b_iq1, b_iq2])

        A_eq = sp.bmat([[A_eq1_u,  A_eq1_x,  None,    None],
                        [None,     A_eq2_x,  A_eq2_a, A_eq2_b]])
        b_eq = np.hstack([b_eq1, b_eq2])

        #############
        # Solve it! #
        #############

        sol = solve_mip(np.zeros(N_tot), A_iq, b_iq, A_eq, b_eq)

        if sol['status'] == 2:
            self.u = np.array(sol['x'][0:N_u]).reshape(T, K * M).transpose()
            self.x = np.hstack([
                np.array(init).reshape(len(init), 1),
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
            self.system_matrix().dot(self.u)
        )

        # Check control constraints
        np.testing.assert_almost_equal(
            self.x[:, :-1],
            self._id_stacked().dot(self.u)
        )

        # Check prefix counting bounds
        for X, R in self.constraints:
            for u in self.u.transpose():
                assert(sum(u[self.order_fcn(v) + m * self.K]
                           for v in self.G.nodes_iter()
                           for m in range(self.M)
                           if (v, m) in X) <= R
                       )

        # Check prefix-suffix connection
        assgn_sum = np.zeros(self.K)
        for C, a in zip(self.cycle_set, self.assignments):
            for (Ci, ai) in zip(C, a):
                assgn_sum[self.order_fcn(Ci[0])] += ai

        np.testing.assert_almost_equal(
            self.x[:, -1],
            assgn_sum
        )

        # Check suffix counting bounds
        ass_rot = [deque(a) for a in self.assignments]
        for X, R in self.constraints:
            for t in range(1000):  # enough to go up to LCM(cycle lengths)
                assert(sum(np.inner(self.cycle_row(C, X), a)
                           for C, a in zip(self.cycle_set, ass_rot)
                           ) <= R)
                for a in ass_rot:
                    a.rotate()
