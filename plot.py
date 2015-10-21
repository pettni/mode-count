import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

filename = 'example_tcl_plot_10_1_3400_10000.save'
filename_hist = 'example_tcl_hist_10_1_3400_10000.save'

max_t = 23.6
min_t = 21.4
num_bin_x = 100
tau = 0.05 
step = (23.6-21.4)/num_bin_x
bins = np.arange(min_t, max_t, step)
tvec = np.arange(0,30,tau/10)

if os.path.isfile(filename_hist):
	hist_vec = pickle.load(open(filename_hist, 'rb'))
else:
	xvec_disc, uvec_disc, xvec_cont, modecount_cont = pickle.load( open(filename, 'rb') )

	hist_vec = []

	for t in range(0, len(xvec_cont)):
		print t
		hist = np.zeros(len(bins))
		for state_i in xvec_cont[t]:
			bin_index = next(i for i in range(len(bins)) if bins[i] + step/2 >= state_i[0] )
			hist[bin_index] += 1
		hist_vec.append(hist)

	hist_vec = np.array(hist_vec)

	pickle.dump(hist_vec, open(filename_hist, 'wb'))


#Direct input 
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 

fig = plt.pcolormesh(tvec[range(0, len(tvec)/3-1, 10)], bins, hist_vec[range(0, len(tvec)/3-1, 10),:].transpose(), cmap = plt.cm.Blues)
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')

plt.savefig('density.eps', format='eps')