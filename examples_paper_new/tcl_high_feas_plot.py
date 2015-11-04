max_t = 23.6
min_t = 21.4
num_bin_x = 200
tau = 0.05 
step = (23.6-21.4)/num_bin_x
bins = np.arange(min_t, max_t, step)
tvec = np.arange(0,30,tau)

if os.path.isfile(filename_hist):
	print "loading histogram"
	hist_vec = pickle.load(open(filename_hist, 'rb'))
else:
	hist_vec = []

	for t in range(0, len(xvec_cont)):
		print t
		hist = np.zeros(len(bins))
		for state_i in xvec_cont[t]:
			bin_index = next(i for i in range(len(bins)) if bins[i] + step/2 >= state_i[0] )
			hist[bin_index] += 1
		hist_vec.append(hist)

	hist_vec = np.array(hist_vec)

	pickle.dump(hist_vec, open(filename_hist, 'wb'))

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

plt.pcolormesh(tvec[range(0, len(tvec)/3-1)], bins, hist_vec[range(0, len(tvec)/3-1),:].transpose(), cmap = plt.cm.Blues)
plt.ylim([min_t, max_t])
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.rcParams.update(params) 
plt.tight_layout()
plt.savefig('tcl_high_feas_density.pdf')
