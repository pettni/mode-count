import sys
import dill
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

sys.path.append('../')
from modecount_new import *
from abstraction import *

for target in ["low", "high"]:

    # Load saved stuff
    if target == "low":
        s1, s2 = dill.load(open("example_tcl_sim_low.p", "r"))
    else:
        s1, s2 = dill.load(open("example_tcl_sim_high.p", "r"))

    xvec1_cont = s1.flatten('F')
    xvec2_cont = s2.flatten('F')
    tvec_cont = [0.05*t for t in range(s1.shape[1]) for n in range(10000)]
    kde = scipy.stats.gaussian_kde(
        np.hstack([np.vstack([tvec_cont, xvec1_cont]),
                   np.vstack([tvec_cont, xvec2_cont])])
    )

    X, Y = np.mgrid[0:0.05*s1.shape[1]:0.05, 21:24:0.05]
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
    axes = plt.gca()
    axes.set_xlim([0,5])

    if target == "low":
        plt.savefig('example_tcl_fig1_low.pdf')
    else:
        plt.savefig('example_tcl_fig1_high.pdf')

    plt.clf()
