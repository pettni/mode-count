import sys
import dill
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

sys.path.append('../')
from modecount_new import *
from abstraction import *

# Load saved stuff
s1, s2 = dill.load(open("example_tcl_sim.p", "r"))

xvec_cont = s1.flatten('F')
tvec_cont = [0.05*t for n in range(10000) for t in range(s1.shape[1])]
kde = scipy.stats.gaussian_kde(np.vstack([tvec_cont, xvec_cont]))

X, Y = np.mgrid[0:0.05*s1.shape[1]:100j, 21:24:30j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True
}

plt.pcolormesh(X,Y,Z, cmap=plt.cm.PuBu)
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.rcParams.update(params) 
plt.tight_layout()

plt.show()