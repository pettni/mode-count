import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

tmin = 0
tmax = 10
tau = 0.05
_, _, _, modecount_low  = pickle.load( open('tcl_low_feas_simulation_20_1_[3200, 3200, 2500, 4600]_10000.save', 'rb') )
_, _, _, modecount_high = pickle.load( open('tcl_high_feas_simulation_20_1_[3600, 3600, 2500, 4600]_10000.save', 'rb') )

tvec = np.arange(0., 10., 0.05)
plt.plot(tvec, np.array(modecount_low)[:len(tvec)], linestyle='red')
plt.plot(tvec, np.array(modecount_high)[:len(tvec)], linestyle='blue')

plt.plot([tmin, tmax], [3200, 3200], linestyle='dashed', color='green')
plt.plot([tmin, tmax], [3600, 3600], linestyle='dashed', color='green')

columnwidth = 240.0/72.27
width = columnwidth

fig = plt.figure(figsize = (width,width/2))
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True
}

h1 = plt.plot(tvec, np.array(modecount_low)[:len(tvec)], color='red')
h2 = plt.plot(tvec, np.array(modecount_high)[:len(tvec)], color='blue')

plt.plot([tmin, tmax], [3200, 3200], linestyle='dashed', color='green')
plt.plot([tmin, tmax], [3600, 3600], linestyle='dashed', color='green')

plt.xlim([tmin, tmax])
plt.ylim([2400, 4700])

plt.xlabel(r'$t$')
plt.ylabel(r'mode-\verb+on+-count')
plt.legend()
plt.rcParams.update(params) 
plt.tight_layout()
plt.savefig('example_5.2_fig2.pdf')
