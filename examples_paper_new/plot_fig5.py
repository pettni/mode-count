import cPickle as pickle

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

X,Y,Z = pickle.load(open('kde_high_feas.save', 'rb'))
X,Y,Z = pickle.load(open('kde_low_feas.save', 'rb'))

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

plt.pcolormesh(X,Y,Z, cmap=plt.cm.YlOrRd)
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.rcParams.update(params) 
plt.tight_layout()
plt.savefig('tcl_high_feas_density.pdf')
