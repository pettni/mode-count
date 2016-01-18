.. currentmodule:: modecount

***************
Documentation
***************

This software package implements methods from the paper "Control synthesis for large collections of systems with mode-counting constraints", which synthesize a control strategy for aggregate dynamics such that mode-counting constraints are satisfied. 

Currently, only mode-counting constraints pertaining to a single mode can be enforced, while the paper above develops the theory for arbitrarily many modes.

Dependencies
===================

* Python with working `numpy <http://www.numpy.org>`_, `scipy <http://www.scipy.org>`_, and `networkx <https://networkx.github.io>`_
* `matplotlib <http://matplotlib.org>`_ (for plotting)
* `Gurobi <http://www.gurobi.com>`_  (recommended) or `Mosek <https://www.mosek.com/products/mosek>`_ and their Python interfaces.

Installation
===================
Currently, no automatic installation is provided. Execute files from the downloaded directory.

Synthesis
===========

The synthesis procedure consists of two steps. First, an abstraction of the continuous dynamics is created. Secondly, a linear program is formed using the abstraction from whose solution an strategy can be extracted.

Construct an abstraction
--------------------------

There is a class :py:class:`Abstraction` for handling abstractions.
Dynamical modes are added to the abstraction using the function :py:func:add_mode`, 
which takes as argument a function representing a vector field::

	# Create an abstraction of the 2D domain [-1,1]^2,
	# with space and time discretization 0.1
	ab = Abstraction([-1, -1], [1, 1], 0.1, 0.1)

	# Add a mode given by the vector field
	# \dot x_0 = -x_1
	# \dot x_1 =  x_0
	ab.add_mode( lambda x: [ -x[1], x[0] ] )

	ab.plot_planar() # plot it

.. autoclass:: Abstraction
	:noindex:

In order to verify that a given abstraction is an approximate bisimulation of a given vector field :math:`\dot x = f(x)`, a :math:`\mathcal{KL}`-function :math:`\beta(\cdot, \cdot)` corresponding to :math:`f` is required.

.. autofunction:: verify_bisim

The :py:class:`Abstraction` has a member *graph* which is a `networkx <https://networkx.github.io>`_ ``DiGraph``,
where edges are labeled according to dynamical mode.


Synthesize a mode-counting strategy
------------------------------------

Given a `networkx <https://networkx.github.io>`_ ``DiGraph``, where edges are labeled with integers numbered from 
1 to :math:`M`, a mode-counting strategy can be synthesized using :py:func:`prefix_suffix_feasible`. A number of 
options are possible, including whether the system should be solved as a linear program (LP), or an integer
linear program (ILP). 

.. autofunction:: prefix_suffix_feasible

::

	# G a graph, `init` required initial condition

	prob_data = {}
	prob_data['graph'] = G				 		   # graph
	prob_data['order_function'] = G.nodes().index  # function mapping node -> integer (improves speed if defined)

	prob_data['init'] = init 					   # initial state configuration
	prob_data['forbidden_nodes'] = G.nodes_with_selfloops() 	# unsafe set

	prob_data['mode'] = 1 						# mode to control
	prob_data['lb_suffix'] = 33 				# lower mode-count bound for mode 1
	prob_data['ub_suffix'] = 33 				# upper mode-count bound for mode 1

	prob_data['ilp'] = True						# solve as ILP

	prob_data['horizon'] = 10 			 		# prefix horizon

	prob_data['cycle_set'] = list(nx.simple_cycles(ab.graph))  # use all simple cycles

	# Solve discrete mode synthesis problem
	prob_sol = prefix_suffix_feasible(prob_data, verbosity=2)

Order function
^^^^^^^^^^^^^^
For computational efficiency, it is beneficial if the index of a nodes in the graph can be indexed at a low computational cost. To this end, an *order function* can be provided as input to many of the functions. An order function to a graph **G** is a function that assigns a unique integer to each node in **G**. For graphs obtained from abstractions as above, an efficient order function is provided in the :py:class:`Abstraction` object::

	# ab is an abstraction
	order_fcn = ab.node_to_idx  # efficient order function for ab.graph

The default behavior is to use ``G.nodes().index``, which must iterate through all nodes of **G** to find the index of a node.

Two-step approach
-----------------

To alleviate the computational burden required by solving an ILP, the following two-step approach can be used

1. Find a non-integer strategy
2. Round the suffix part to integers
3. Find a prefix to the suffix

The suffix part of a non-integer strategy resulting from an LP can be rouded using :py:func:`make_integer`. Then, :py:func:`prefix_feasible` can be called to solve a (significantly smaller) ILP in order to find a 
prefix strategy.::

	# Set up and solve LP
	nonint_data = { ..., ilp = False, ... }
	nonint_solution = prefix_suffix_feasible(nonint_data)

	# Set up and solve ILP
	int_data = nonint_data.copy()
	int_data['ilp'] = True
	int_data['cycle_set'] = nonint_solution['cycles']  
	# round the suffix part of LP solution
	int_data['assignments'] = make_integer(nonint_solution['assignments'])

	# Find a prefix
	int_solution = prefix_feasible(nonint_data)

.. autofunction:: prefix_feasible

Simulate strategy
==================

A strategy synthesized as above can be simulated with an instance of :py:class:`CycleControl`, which provides for a linear system :math:`\mathbf{w}(t+1) + A \mathbf{w} + B \mathbf{r}`. Such matrices :math:`A,B` can be computed by :py:func:`lin_syst` (note that the same order function must be used throughout).::
	
	# assume an abstraction ab has been created
	problem_data = { ..., 'graph' = ab.graph, 'order_function' = ab.node_to_idx, ... }
	int_solution = prefix_suffix_feasible(problem_data)
	A,B = lin_syst(ab.graph, order_fcn = ab.node_to_idx)

	controls = CycleControl(ab.graph, int_solution, ab.node_to_idx)

	# controls can now be extracted with controls.get_u(t, state)

Example
========
Working examples can be found in the ``examples/`` and ``examples_paper/`` directories.