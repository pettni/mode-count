import dill
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde
import numpy as np

for target in ["low", "high"]:

    # Load saved stuff
    if target == "low":
        s1, s2, m1, m2 = dill.load(open("saved/example_tcl_sim_low.p", "r"))
    else:
        s1, s2, m1, m2 = dill.load(open("saved/example_tcl_sim_high.p", "r"))

    xvec1_cont = s1.flatten('F')
    xvec2_cont = s2.flatten('F')
    tvec_cont = [0.05*t for t in range(s1.shape[1]) for n in range(10000)]
    kde = gaussian_kde(
        np.hstack([np.vstack([tvec_cont, xvec1_cont]),
                   np.vstack([tvec_cont, xvec2_cont])])
    )

    X, Y = np.mgrid[0:0.05*s1.shape[1]:0.05, 21:24:0.05]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    fig = plt.figure()
    fig.set_size_inches(6, 4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim([0,5])
    ax.set_ylim([21,24])
    
    fig.add_axes(ax)

    if target == "low":
        ax.pcolormesh(X,Y,Z, cmap=plt.cm.PuBu)
        plt.savefig('saved/example_tcl_fig1_low.pdf', bbox_inches=0)
    else:
        ax.pcolormesh(X,Y,Z, cmap=plt.cm.PuRd)
        plt.savefig('saved/example_tcl_fig1_high.pdf', bbox_inches=0)

    plt.clf()
