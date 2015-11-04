filename_simulation = ''

############################################################################
##################### Perform the simulation ###############################
############################################################################

if os.path.isfile(filename_simulation):
	print "loading saved simulation"
	xvec_disc, uvec_disc, xvec_cont, modecount_cont = pickle.load( open(filename_simulation, 'rb') )
else:
	A,B = lin_syst(G, order_fcn)
	controls = CycleControl(G, mc_sol_int, order_fcn)

	disc_init = mc_sol_int['states'][:,0]

	tmax = 10
	disc_tmax = int(tmax / tau)

	xvec_disc = np.zeros([len(disc_init), disc_tmax+1], dtype=float)
	uvec_disc = np.zeros([len(disc_init), disc_tmax+1], dtype=float)
	xvec_disc[:,0] = disc_init

	xvec_cont = []
	modecount_cont = []

	# set TCL discrete state and create lists
	# for easy access to "all TCLs in a certain node"
	vertex_to_tcls = [ [] for i in range(len(G)) ]
	for i, tcl in enumerate(population):
		tcl.disc_state = order_fcn( ab.point_to_midx(tcl.state) )
		vertex_to_tcls[tcl.disc_state].append(i)

	for disc_t in range(disc_tmax):
		print "TIME: ", disc_t, " out of ", disc_tmax

		u = controls.get_u(disc_t, xvec_disc[:,disc_t])

		flows = xvec_disc[:,disc_t] - u + scipy.sparse.bmat( [ [None, scipy.sparse.identity(len(G))], [scipy.sparse.identity(len(G)), None] ] ).dot( u )

		# assign modes
		for i, tcls in enumerate(vertex_to_tcls):
			num_mode_on = round(flows[i])
			num_mode_off = round(flows[i+len(G)])

			if num_mode_on:
				assert len( [new_node for new_node in G.successors( ab.idx_to_node(i) ) if G[ab.idx_to_node(i)][new_node]['mode'] == 1] ) >= 1 
			if num_mode_off:
				assert len( [new_node for new_node in G.successors( ab.idx_to_node(i) ) if G[ab.idx_to_node(i)][new_node]['mode'] == 2] ) >= 1

			for j, tcl_index in enumerate(tcls):
				if j < num_mode_on:
					population[tcl_index].mode = 'on'
				else:
					population[tcl_index].mode = 'off'

		# simulate continuous time
		step = 1
		for i in range(step):
			dt = tau/step
			for tcl in population: tcl.step(dt)
			xvec_cont.append(np.array([ tcl.state for tcl in population ]))
			modecount_cont.append( sum( [int(tcl.mode == 'on') for tcl in population]  ) )

		# update discrete time
		for i, tcl in enumerate(population):
			current_node = ab.idx_to_node(tcl.disc_state)
			mode_num = 1 if tcl.mode == 'on' else 2
			next_disc_state = order_fcn([next_node for next_node in G.successors(current_node) if G[current_node][next_node]['mode'] == mode_num][0])

			vertex_to_tcls[tcl.disc_state].remove(i)
			vertex_to_tcls[next_disc_state].append(i)

			tcl.disc_state = next_disc_state

		uvec_disc[:,disc_t] = u
		xvec_disc[:,disc_t+1] = A.dot(xvec_disc[:,disc_t]) + B.dot(u)

		assert( abs(sum(xvec_disc[:, disc_t+1]) - 10000) < 1e-5)

	pickle.dump((xvec_disc, uvec_disc, xvec_cont, modecount_cont), open(filename_simulation, 'wb') )