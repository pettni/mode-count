import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import product

sys.path.append('../')
from modecount_new import *
from abstraction import *
from random_cycle_new import random_cycle

# Set 1 of TCL parameters
Cth_1 = 2.
Rth_1 = 2.
Pm_1 = 5.6
eta_tcl_1 = 2.5
pop_size_1 = 15

# Set 2 of TCL parameters
Cth_2 = 2.2
Rth_2 = 2.2
Pm_2 = 5.9
eta_tcl_2 = 2.5
pop_size_2 = 15

# Ambient temperature
theta_a = 32.

# Abstraction parameters
eta_1 = 0.002               # space discretization
tau_1 = 0.05                  # time discretization
eps_1 = 0.2                   # desired bisimulation approximation

eta_2 = 0.0015               # space discretization
tau_2 = 0.05                  # time discretization
eps_2 = 0.2                   # desired bisimulation approximation

# Disturbance level (same for both)
delta_vf = 0.025  # disturbance level

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

# eps_r = np.linspace(0, 0.2, 100)
# plt.plot(eps_r, [e - beta_on_1(e, tau) - eta / 2 -
# (delta_vf / K) * (np.exp(K * tau) - 1) for e in eps_r])
# plt.show()

# Make sure that bisimilarity holds
assert(verify_bisim(beta_on_1, tau_1, eta_1, eps_1, delta_vf, K_1))
assert(verify_bisim(beta_off_1, tau_2, eta_2, eps_2, delta_vf, K_2))
assert(verify_bisim(beta_on_2, tau_2, eta_2, eps_2, delta_vf, K_2))
assert(verify_bisim(beta_off_2, tau_2, eta_2, eps_2, delta_vf, K_2))

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
# Load or compute abstractions #
################################

print "computing first abstraction"
print "abstraction will have ", \
      np.product((np.array(ub) - np.array(lb)) / eta_1), " states"
ab1 = Abstraction(lb, ub, eta_1, tau_1)
ab1.add_mode(vf_on_1, 'on')
ab1.add_mode(vf_off_1, 'off')

G1 = ab1.graph

print "computing second abstraction"
print "abstraction will have ", \
      np.product((np.array(ub) - np.array(lb)) / eta_2), " states"
ab2 = Abstraction(lb, ub, eta_2, tau_2)
ab2.add_mode(vf_on_2, 'on')
ab2.add_mode(vf_off_2, 'off')

G2 = ab2.graph

##########################
# Generate random cycles #
##########################


# Augment a cycle with first outgoing mode
def augment(G, c):
    outg = [G[c[i]][c[(i + 1) % len(c)]]['modes'][0]
            for i in range(len(c))]
    return zip(c, outg)


cycle_set1 = []
c_quot_set = set([])
while len(cycle_set1) < 25:
    c = random_cycle(G1, [], 5, 0.8)
    c = augment(G1, c)
    c_quot = float(sum(1 for ci in c if 'on' in ci[1])) / len(c)
    if c_quot not in c_quot_set:
        cycle_set1.append(c)
        c_quot_set.add(c_quot)

cycle_set2 = []
c_quot_set = set([])
while len(cycle_set2) < 25:
    c = random_cycle(G2, [], 5, 0.8)
    c = augment(G2, c)
    c_quot = float(sum(1 for ci in c if 'on' in ci[1])) / len(c)
    if c_quot not in c_quot_set:
        cycle_set2.append(c)
        c_quot_set.add(c_quot)


###########################
# Set up counting problem #
###########################

state_1 = [22 + 1 * np.random.rand(1) for i in range(pop_size_1)]
state_2 = [22 + 1 * np.random.rand(1) for i in range(pop_size_2)]

cp = MultiCountingProblem(2)
# Discrete structures from abstractions
cp.graphs[0] = G1
cp.graphs[1] = G2

# Initial aggregate conditions
cp.inits[0] = np.zeros(len(G1))
for s in state_1:
    cp.inits[0][G1.order_fcn(ab1.point_to_midx(s))] += 1
cp.inits[1] = np.zeros(len(G2))
for s in state_2:
    cp.inits[1][G2.order_fcn(ab2.point_to_midx(s))] += 1

# Cycle sets
cp.cycle_sets[0] = cycle_set1
cp.cycle_sets[1] = cycle_set2

# Add counting constraints
cc1 = CountingConstraint(2)  # mode counting
cc1.X[0] = set(product(G1.nodes(), ['on']))
cc1.X[1] = set(product(G2.nodes(), ['on']))
cc1.R = (pop_size_1 + pop_size_2) / 2

cc2 = CountingConstraint(2)  # safety
unsafe_1 = [v for v, d in G1.nodes_iter(data=True)
            if d['mid'] > theta_r + delta_th or
            d['mid'] < theta_r - delta_th]
cc2.X[0] = set(product(unsafe_1, ['on', 'off']))
unsafe_2 = [v for v, d in G2.nodes_iter(data=True)
            if d['mid'] > theta_r + delta_th or
            d['mid'] < theta_r - delta_th]
cc2.X[1] = set(product(unsafe_2, ['on', 'off']))
cp.constraints += [cc1, cc2]

# Problem horizon
cp.T = 10
cp.solve_prefix_suffix()
