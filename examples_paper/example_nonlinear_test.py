import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import numpy.lib.scimath

# Parameters
g = 9.8 # gravitational coefficient 	m/s^2
k = 5.  # friction coefficient 		N/m = kg / s^2
m = 0.5 # pendulum mass				kg
l = 1.  # pendulum arm length 		m

umax = 5

tau = 0.15

l1 = -k/(2*m) + np.lib.scimath.sqrt( k**2/(4*m**2) - np.sqrt( g**2/l**2 - umax**2 ) )
l2 = -k/(2*m) - np.lib.scimath.sqrt( k**2/(4*m**2) - np.sqrt( g**2/l**2 - umax**2 ) )
eq = np.arcsin(l*umax/g)

print "eigenvalues", l1, l2
print "equilibrium", eq

# Define a vector fields
vf1 = lambda x, t : [x[1], -(g/l) * np.sin(x[0]) - (k/m) * np.sin(x[1]) + umax]
vf2 = lambda x, t : [x[1], -(g/l) * np.sin(x[0]) - (k/m) * np.sin(x[1]) - umax]

for xinit in np.arange(-1,1,0.1):
	for yinit in np.arange(-1,1,0.1):
		sol = scipy.integrate.odeint(vf1, [xinit, yinit], np.arange(0, tau, tau/100))
		plt.plot(sol[:,0], sol[:,1], 'blue')

for xinit in np.arange(-1,1,0.1):
	for yinit in np.arange(-1,1,0.1):
		sol = scipy.integrate.odeint(vf2, [xinit, yinit], np.arange(0, tau, tau/100))
		plt.plot(sol[:,0], sol[:,1], 'red')

plt.show()
