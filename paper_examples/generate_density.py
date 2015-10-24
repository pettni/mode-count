import cPickle as pickle

import numpy as np
import scipy.stats

tau = 0.05
popsize = 10000

tmin = 0.
tmax = 200*tau
xmin = 21.4
xmax = 23.6

# _, _, xvec_cont, _ = pickle.load( open('tcl_low_feas_plotdata_20_1_[3210, 3190]_10000.save', 'rb') )
_, _, xvec_cont, _ = pickle.load( open('tcl_high_feas_plotdata_20_1_[3610, 3590]_10000.save', 'rb') )

xvec_cont = np.array(xvec_cont).flatten()
tvec_cont = np.array([([n * tau] * popsize) for n in range(200)]).flatten()

kde = scipy.stats.gaussian_kde(np.vstack([tvec_cont, xvec_cont]))

X, Y = np.mgrid[tmin:tmax:100j, xmin:xmax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

pickle.dump((X,Y,Z), open('kde_high_feas.save', 'wb'))