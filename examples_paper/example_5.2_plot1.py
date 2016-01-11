import cPickle as pickle

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

max_t = 23.6
min_t = 21.4
num_bin_x = 100
tau = 0.05 
step = (23.6-21.4)/num_bin_x
bins = np.arange(min_t, max_t, step)
tvec = np.arange(0,30,tau/10)

for target in ['low', 'high']:

	if target == 'low':
		_, _, xvec_cont, _ = pickle.load( open('tcl_low_feas_simulation_20_1_[3200, 3200, 2500, 4600]_10000.save', 'rb') )
	elif target == 'high':
		_, _, xvec_cont, _ = pickle.load( open('tcl_high_feas_simulation_20_1_[3600, 3600, 2500, 4600]_10000.save', 'rb') )


	hist_vec = []

	for t in range(0, len(xvec_cont)):
		print t
		hist = np.zeros(len(bins))
		for state_i in xvec_cont[t]:
			bin_index = next(i for i in range(len(bins)) if bins[i] + step/2 >= state_i[0] )
			hist[bin_index] += 1
		hist_vec.append(hist)

	hist_vec = np.array(hist_vec)


	#Direct input 
	plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
	#Options
	params = {'text.usetex' : True,
	          'font.size' : 11,
	          'font.family' : 'lmodern',
	          'text.latex.unicode': True,
	          }
	plt.rcParams.update(params) 

	fig = plt.pcolormesh(tvec[range(0, len(tvec)/3, 10)], bins, hist_vec.transpose(), cmap = plt.cm.Blues)
	plt.xlabel(r'$t$')
	plt.ylabel(r'$\theta$')

	if target == 'low':
		plt.savefig('example_5.2_fig1a_nosmooth.pdf')
	elif target == 'high':
		plt.savefig('example_5.2_fig1b_nosmooth.pdf')

