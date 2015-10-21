import numpy as np

def tcl_model():

	# TCL parameters
	Cth = 2.
	Rth = 2.
	Pm = 5.6
	eta_tcl = 2.5

	# Ambient temperature
	theta_a = 32.

	# Derived constants
	a = 1./(Rth * Cth)
	b = eta_tcl / Cth

	# Define a vector fields
	vf1 = lambda theta : -a * ( theta - theta_a ) - b * Pm  # tcl on
	vf2 = lambda theta : -a * ( theta - theta_a ) 			# tcl off
	
	# Define a KL function beta(r,s) s.t. || phi(t,x) - phi(t,y) || <= beta(||x-y||, t)
	kl1 = lambda r,s : r * np.exp(-s*a)
	kl2 = lambda r,s : r * np.exp(-s*a)

	return vf1, vf2, kl1, kl2