
def casc_round(list):
	"""
		Round entries in list while preserving total sum:
		i_n = round(f_0 + ... + f_n) - (i_0 + ... + i_n-1)
	"""
	fp_total = 0.
	int_total = 0

	ret = [0] * len(list)

	for i,fp in enumerate(list):
		ret[i] = round(fp_total + fp) - int_total
		fp_total += fp
		int_total += ret[i]

	assert(round(sum(list)) == round(sum(ret)) )

	return ret

def make_integer(assignments):

	ass_sums = [sum(a) for a in assignments]
	total_sum = sum([sum(a) for a in assignments])

	rounded_sums = casc_round(ass_sums)

	for ass, int_sum in zip(assignments, rounded_sums):
		diff = int_sum - sum(ass)
		for j in range(len(ass)):
			ass[j] = ass[j] + diff/len(ass)

	return [casc_round(ass) for ass in assignments]

