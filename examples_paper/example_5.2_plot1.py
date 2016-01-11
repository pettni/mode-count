import cPickle as pickle

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

tau = 0.05
popsize = 10000

tmin = 0.
tmax = 200*tau
xmin = 21.4
xmax = 23.6

width = 240.0/72.27

for target in ['low', 'high']:

	if target == 'low':
		_, _, xvec_cont, _ = pickle.load( open('tcl_low_feas_simulation_20_1_[3200, 3200, 2500, 4600]_10000.save', 'rb') )
	elif target == 'high':
		_, _, xvec_cont, _ = pickle.load( open('tcl_high_feas_simulation_20_1_[3600, 3600, 2500, 4600]_10000.save', 'rb') )


	xvec_cont = np.array(xvec_cont).flatten()
	tvec_cont = np.array([([n * tau] * popsize) for n in range(200)]).flatten()
	kde = scipy.stats.gaussian_kde(np.vstack([tvec_cont, xvec_cont]))
	X, Y = np.mgrid[tmin:tmax:100j, xmin:xmax:100j]
	positions = np.vstack([X.ravel(), Y.ravel()])
	Z = np.reshape(kde(positions).T, X.shape)


	fig = plt.figure(figsize = (width,width/2))

	plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
	#Options
	params = {'text.usetex' : True,
	          'font.size' : 11,
	          'font.family' : 'lmodern',
	          'text.latex.unicode': True
	}

	plt.pcolormesh(X,Y,Z, cmap=plt.cm.YlOrRd)
	plt.xlabel(r'$t$')
	plt.ylabel(r'$\theta$')
	plt.rcParams.update(params) 
	plt.tight_layout()
	if target == 'low':
		plt.savefig('example_5.2_fig1a.pdf')
	elif target == 'high':
		plt.savefig('example_5.2_fig1b.pdf')
		

