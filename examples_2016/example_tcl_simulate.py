import sys
import dill

sys.path.append('../')
from modecount_new import *
from abstraction import *

# Time steps to simulate
T = 100

# Load saved stuff
state_1, state_2, ab1, ab2, params_1, params_2, tau, cp = dill.load(open("example_tcl_sol.p", "r"))

# Constant model errors
dist_1 = -0.02 + 0.04 * np.random.rand(len(state_1))
dist_2 = -0.02 + 0.04 * np.random.rand(len(state_2))


# Get parameters
a_1, b_1_on, b_1_off = params_1
a_2, b_2_on, b_2_off = params_2


# Discrete state
disc_state = [[ab1.point_to_midx(s) for s in state_1],
			  [ab2.point_to_midx(s) for s in state_2]]

# Continuous state
s1 = np.zeros([len(state_1), T])
s2 = np.zeros([len(state_1), T])
s1[:,0] = state_1
s2[:,0] = state_2


for t in range(T-1):
	
	actions = cp.get_input(disc_state, t)

	# On/off offsets
	b_vec1 = np.array([b_1_on if act == 'on' else b_1_off for act in actions[0]])
	b_vec2 = np.array([b_2_on if act == 'on' else b_2_off for act in actions[1]])

	# New continuous states
	s1[:, t+1] = np.exp(tau * a_1) * s1[:, t] + (b_vec1/a_1) * (np.exp(tau * a_1) - 1) + tau * dist_1
	s2[:, t+1] = np.exp(tau * a_2) * s2[:, t] + (b_vec2/a_2) * (np.exp(tau * a_2) - 1) + tau * dist_2

	# Update discrete states
	for i in range(10000):
		disc_state[0][i] = ab1.graph.post(disc_state[0][i], actions[0][i])
		disc_state[1][i] = ab2.graph.post(disc_state[1][i], actions[1][i])

dill.dump([s1, s2], open( "example_tcl_sim.p", "wb"))