def linear_transform(m: list[list[int]], v: list[int]):
	out = [0] * len(v)

	for i, dim in enumerate(m):
		for j, basis_vec in enumerate(dim):
			out[i] += basis_vec * v[j]

	return out


# takes l0 and weights_l1 to return l1
def propagate(weights_l1: list[list[int]], activations_l0: list[int]):  # linear tranformation from the light of neurons activations
	activations_l1 = [0]*len(weights_l1)

	for i, a_l0 in enumerate(weights_l1):  # for each list of weights of neurons of l0
		for j, w_ail0_ajl1 in enumerate(a_l0):  # for each weight from a_l0_i to a_
			activations_l1[i] += w_ail0_ajl1 * activations_l0[j]

	return activations_l1
