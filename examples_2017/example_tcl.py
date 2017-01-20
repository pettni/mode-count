import numpy as np
import sys
import dill
from itertools import product

sys.path.append('../')
from modecount import *
from abstraction import *
from random_cycle import random_cycle

# Upper or lower counting bound
experiment = "high"

# Set 1 of TCL parameters
Cth_1 = 2.
Rth_1 = 2.
Pm_1 = 5.6
eta_tcl_1 = 2.5
pop_size_1 = 10000

# Set 2 of TCL parameters
Cth_2 = 2.2
Rth_2 = 2.2
Pm_2 = 5.9
eta_tcl_2 = 2.5
pop_size_2 = 10000

# Ambient temperature
theta_a = 32.

# Abstraction parameters
tau = 0.05                  # time discretization

eta_1 = 0.002               # space discretization
eps_1 = 0.2                   # desired bisimulation approximation

eta_2 = 0.0015               # space discretization
eps_2 = 0.2                   # desired bisimulation approximation

# Disturbance level (same for both)
delta_vf = 0.025  # disturbance level

# Prefix horizon
horizon = 20

# Cycle count
num_cycles = 50

# Simulation horizon
T = 110

# Derived constants
a_1 = 1. / (Rth_1 * Cth_1)
b_1 = eta_tcl_1 / Cth_1
a_2 = 1. / (Rth_2 * Cth_2)
b_2 = eta_tcl_2 / Cth_2

# Vector field Lipschitz constants
K_1 = a_1
K_2 = a_2

# Define vector fields
vf_on_1 = lambda theta: -a_1 * (theta - theta_a) - b_1 * Pm_1
vf_off_1 = lambda theta: -a_1 * (theta - theta_a)
vf_on_2 = lambda theta: -a_2 * (theta - theta_a) - b_2 * Pm_2
vf_off_2 = lambda theta: -a_2 * (theta - theta_a)

# Define a KL function beta(r,s) s.t.
# || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
beta_on_1 = lambda r, s: r * np.exp(-s * a_1)
beta_off_1 = lambda r, s: r * np.exp(-s * a_1)
beta_on_2 = lambda r, s: r * np.exp(-s * a_2)
beta_off_2 = lambda r, s: r * np.exp(-s * a_2)

# Make sure that bisimilarity holds
assert(verify_bisim(beta_on_1, tau, eta_1, eps_1, delta_vf, K_1))
assert(verify_bisim(beta_off_1, tau, eta_1, eps_1, delta_vf, K_1))
assert(verify_bisim(beta_on_2, tau, eta_2, eps_2, delta_vf, K_2))
assert(verify_bisim(beta_off_2, tau, eta_2, eps_2, delta_vf, K_2))

################################################
################################################
################################################

# desired temperature interval
theta_r = 22.5
delta_th = 1.2

# Abstraction parameters
lb = [theta_r - delta_th]      # lower bounds
ub = [theta_r + delta_th]      # upper bounds

# set random seed for repeatability
np.random.seed(0)

################################

# Print analytical bounds
print "Maximal lower bound:", \
    pop_size_1 * vf_off_1(lb[0]) / (-vf_on_1(lb[0]) + vf_off_1(lb[0])) \
    + pop_size_2 * vf_off_2(lb[0]) / (-vf_on_2(lb[0]) + vf_off_2(lb[0]))

print "Mimimal upper bound:", \
    pop_size_1 + pop_size_2 - \
    pop_size_1 * -vf_on_1(ub[0]) / (-vf_on_1(ub[0]) + vf_off_1(ub[0])) \
    - pop_size_2 * -vf_on_2(ub[0]) / (-vf_on_2(ub[0]) + vf_off_2(ub[0]))


################################


################################
# Load or compute abstractions #
################################

print "computing first abstraction"
print "abstraction will have ", \
      np.product((np.array(ub) - np.array(lb)) / eta_1), " states"
ab1 = Abstraction(lb, ub, eta_1, tau)
ab1.add_mode(vf_on_1, 'on')
ab1.add_mode(vf_off_1, 'off')

G1 = ab1.graph

print "computing second abstraction"
print "abstraction will have ", \
      np.product((np.array(ub) - np.array(lb)) / eta_2), " states"
ab2 = Abstraction(lb, ub, eta_2, tau)
ab2.add_mode(vf_on_2, 'on')
ab2.add_mode(vf_off_2, 'off')

G2 = ab2.graph

###########################
# Set up counting problem #
###########################

state_1 = [21.6 + 1.8 * np.random.rand(1) for i in range(pop_size_1)]
state_2 = [21.6 + 1.8 * np.random.rand(1) for i in range(pop_size_2)]

cp = MultiCountingProblem(2)
# Discrete structures from abstractions
cp.graphs[0] = G1
cp.graphs[1] = G2

# Initial aggregate conditions
cp.inits[0] = np.zeros(len(G1), dtype=int)
for s in state_1:
    cp.inits[0][G1.order_fcn(ab1.point_to_midx(s))] += 1
cp.inits[1] = np.zeros(len(G2), dtype=int)
for s in state_2:
    cp.inits[1][G2.order_fcn(ab2.point_to_midx(s))] += 1

# Add counting constraints
cc1 = CountingConstraint(2)  # mode counting
if experiment == "low":
    cc1.X[0] = set(product(G1.nodes(), ['on']))
    cc1.X[1] = set(product(G2.nodes(), ['on']))
    cc1.R = 6000
else:
    cc1.X[0] = set(product(G1.nodes(), ['off']))
    cc1.X[1] = set(product(G2.nodes(), ['off']))
    cc1.R = 20000 - 6700

cc2 = CountingConstraint(2)  # safety
unsafe_1 = [v for v, d in G1.nodes_iter(data=True)
            if d['mid'] > theta_r + delta_th - eps_1 or
            d['mid'] < theta_r - delta_th + eps_1]
cc2.X[0] = set(product(unsafe_1, ['on', 'off']))
unsafe_2 = [v for v, d in G2.nodes_iter(data=True)
            if d['mid'] > theta_r + delta_th - eps_2 or
            d['mid'] < theta_r - delta_th + eps_2]
cc2.X[1] = set(product(unsafe_2, ['on', 'off']))

cp.constraints += [cc1, cc2]


# Cycle sets
cycle_set1 = []
c_quot_set = set([])
while len(cycle_set1) < num_cycles:
    c = random_cycle(G1, unsafe_1, 5, 0.8)
    c = augment(G1, c)
    c_quot = float(sum(1 for ci in c if 'on' in ci[1])) / len(c)
    if c_quot not in c_quot_set:
        cycle_set1.append(c)
        c_quot_set.add(c_quot)

cycle_set2 = []
c_quot_set = set([])
while len(cycle_set2) < num_cycles:
    c = random_cycle(G2, unsafe_2, 5, 0.8)
    c = augment(G2, c)
    c_quot = float(sum(1 for ci in c if 'on' in ci[1])) / len(c)
    if c_quot not in c_quot_set:
        cycle_set2.append(c)
        c_quot_set.add(c_quot)

cp.cycle_sets[0] = cycle_set1
cp.cycle_sets[1] = cycle_set2


# Problem horizon
cp.T = horizon
print cp.solve_prefix_suffix(solver='mosek', output=False)

cp.test_solution()

##############
# SIMULATION #
##############

# Constant model errors
dist_1 = -0.02 + 0.04 * np.random.rand(len(state_1))
dist_2 = -0.02 + 0.04 * np.random.rand(len(state_2))

# Get parameters
a_1, b_1_on, b_1_off = [vf_on_1(1) - vf_on_1(0), vf_on_1(0), vf_off_1(0)]
a_2, b_2_on, b_2_off = [vf_on_2(1) - vf_on_2(0), vf_on_2(0), vf_off_2(0)]

# Discrete states
disc_state = [[ab1.point_to_midx(s) for s in state_1],
              [ab2.point_to_midx(s) for s in state_2]]

# Continuous state
s1 = np.zeros([len(state_1), T])
s2 = np.zeros([len(state_1), T])
s1[:, 0] = state_1
s2[:, 0] = state_2

for t in range(T - 1):
    actions = cp.get_input(disc_state, t)

    # On/off offsets
    b_vec1 = np.array([b_1_on if act == 'on'
                       else b_1_off for act in actions[0]])
    b_vec2 = np.array([b_2_on if act == 'on'
                       else b_2_off for act in actions[1]])

    # New continuous states
    s1[:, t + 1] = np.exp(tau * a_1) * s1[:, t] + (b_vec1 / a_1) \
        * (np.exp(tau * a_1) - 1) + tau * dist_1
    s2[:, t + 1] = np.exp(tau * a_2) * s2[:, t] + (b_vec2 / a_2) \
        * (np.exp(tau * a_2) - 1) + tau * dist_2

    # Update discrete states
    for i in range(10000):
        disc_state[0][i] = ab1.graph.post(disc_state[0][i], actions[0][i])
        disc_state[1][i] = ab2.graph.post(disc_state[1][i], actions[1][i])

if experiment == "low":
    dill.dump([s1, s2], open("example_tcl_sim_low.p", "wb"))
else:
    dill.dump([s1, s2], open("example_tcl_sim_high.p", "wb"))
