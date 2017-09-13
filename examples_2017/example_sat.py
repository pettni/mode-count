import numpy as np
import sys

sys.path.append('../')
from counting import *
from abstraction import *
from rounding import *
from itertools import product
from random_cycle import random_cycle
from random import choice 

import matplotlib.pyplot as plt

# Dynamics parameters
sigma = 10.
beta = 11./3

# Random seed for reproducibility
seed = 5000

# Dynamics coefficient
coef = 3

####################
# SPECIFY DYNAMICS #
####################

sat = lambda x: np.maximum(-1, np.minimum(1,x))


vf_plus= lambda x: [-beta*x[0] + coef*sat(x[1])x[2]**2, -sigma*x[1] + sigma*x[2], -x[2] + 2];
vf_min = lambda x: [-beta*x[0] + coef*sat(x[1])x[2]**2, -sigma*x[1] + sigma*x[2], -x[2] - 2];

beta1 = lambda r,t: np.exp(-beta*t) * r
gamma1 = lambda r: coef*sat(r)**2/beta

beta2 = lambda r,t: ((sigma*np.exp(-t) - np.exp(-sigma*t))/(sigma - 1)) * r
gamma2 = lambda r: sigma/(sigma-1) * r;

beta_a = lambda r,t: beta1(3*beta1(r,t/2),t/2)
beta_b = lambda r,t: beta1(3*gamma1(2*beta2(r,0)),t/2) + gamma1(2*beta2(r,t/2));

beta_nc = lambda r,t: beta1(beta1(r,t/2) + gamma1(beta2(r,0)),t/2) + gamma1(beta2(r,t/2));

eps = 0.1;
eta = 0.05;
tau = 0.138;

if True:
    # plot curve
    tt = np.arange(0,2,0.01)
    plt.plot(tt, beta_nc(eps, tt) + eta/2)
    plt.plot(tt, eps*np.ones(tt.shape))
    plt.plot(tau, beta_nc(eps, tau) + eta/2, marker='o')
    plt.show()

assert(verify_bisim(beta_nc, tau, eta, eps, 0, 1))

######################
# CREATE ABSTRACTION #
######################

ab = Abstraction([-1,-3,-3], [1,3,3], eta, tau)
print "Adding mode on"
ab.add_mode(vf_plus, 'on')
print "Adding mode off"
ab.add_mode(vf_min, 'off')

###########################
# SET UP COUNTING PROBLEM #
###########################

G = ab.graph

cp = MultiCountingProblem(1)
# Discrete structures from abstractions
cp.graphs[0] = G

# Add counting constraints
cc1 = CountingConstraint(1)  # mode counting
set1 = set([node for node,attr in G.nodes_iter(data=True) if attr['mid'][0]<eps])
cc1.X[0] = set(product(set1, ['on', 'off']))

cc2 = CountingConstraint(1)  # mode counting
set2 = set([node for node,attr in G.nodes_iter(data=True) if attr['mid'][0]>-eps])
cc2.X[0] = set(product(set2, ['on', 'off']))



for frac in [0.6, 0.65, 0.7, 0.8]:
    for cycle_set_size in [100, 200]:
        for horizon in [10, 12, 14, 16]:
            
            np.random.seed(seed)

            cp.inits[0] = np.random.multinomial(10000, [1./len(ab)]*len(ab))

            cc1.R = frac*sum(cp.inits[0])
            cc2.R = frac*sum(cp.inits[0])

            cp.constraints = [cc1, cc2]

            # Cycle sets
            cycle_set = []
            num = 0
            while len(cycle_set) < cycle_set_size:
                print "sampling cycle %d" % num
                c = random_cycle(G, set([]), 2, 0.8)
                num += 1
                if (set(c) - set1) and (set(c) - set2):
            	    c = augment(G, c)
            	    cycle_set.append(c)

            cp.cycle_sets[0] = cycle_set

            # Problem horizon
            cp.T = horizon

            # Solving LP
            print "Solving LP: frac=%f, cc_size=%d, horiz=%d" % (frac, cycle_set_size, horizon)
            cp.solve_prefix_suffix(solver='gurobi', output=True, integer=False)

# nonint_assignments = cp.assignments[0]
# cp.assignments[0] = round_suffix(nonint_assignments)

# print "Solving ILP"
# solve_prefix(cp, solver='gurobi', output=True)

# # Save counting data
# tt = np.arange(50)
# count1 = np.zeros(tt.shape)
# count2 = np.zeros(tt.shape)

# for t in tt:
# 	count1[t] = float(cp.mode_count(cc1.X, t))/sum(cp.inits[0])
# 	count2[t] = float(cp.mode_count(cc2.X, t))/sum(cp.inits[0])

# np.savetxt('data.txt', np.vstack([tt, count1, count2]).transpose(), fmt=['%d', '%0.2f', '%0.2f'], delimiter=',')

# # Save example cycle coordinates
# for cyclenum in range(len(cp.cycle_sets[0])):
    
#     if sum(cp.assignments[0][cyclenum]) > 1:

#         cycle = cp.cycle_sets[0][cyclenum]
#         cycle_coords = np.array([G.node[v]['mid'] for v,_ in cycle])
#         np.savetxt('cycle%d.txt' % cyclenum, cycle_coords, delimiter=',')
