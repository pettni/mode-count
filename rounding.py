import numpy as np

def round_suffix(assignments):

	# List total assignment weights
	Ni_list = [sum(ass) for ass in assignments]

	# Integer total assignment weights
	Ni_int = [int(Ni + 1e-10) for Ni in Ni_list]

	# Extra weight
	extra = int(round(sum(N1 - N2 for N1, N2 in zip(Ni_list, Ni_int))))
	nonzero_idx = np.nonzero(Ni_list)
	for i in range(extra):
		Ni_int[nonzero_idx[0][i]] += 1

	assert(sum(Ni_int) == int(round(sum(Ni_list))))

	new_assignments = [ass for ass in assignments]

	for i in nonzero_idx[0]:

		kappa2 = Ni_int[i] % len(assignments[i])
		kappa1 = (Ni_int[i] - kappa2)/len(assignments[i])

		new_assignments[i] = kappa1*np.ones(len(assignments[i]))

		if kappa2 > 0:
			d = len(assignments[i])/kappa2
			for j in range(kappa2):
				new_assignments[i][np.int(np.floor(j*d))] += 1

		assert(sum(new_assignments[i]) == Ni_int[i])

	return new_assignments
