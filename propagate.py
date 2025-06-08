type vector = list[float]
type matrix = list[list[float]]

from activation import relu


def linear_transform(m: list[list[int]], v: list[int]):
	out = [0] * len(v)

	for i, dim in enumerate(m):
		for j, basis_vec in enumerate(dim):
			out[i] += basis_vec * v[j]

	return out


# takes l0 and weights_l1 to return l1
def propagate_weights(weights_l1: list[list[float]], activations_l0: list[float]):  # linear tranformation from the light of neurons activations
	activations_l1 = [0]*len(weights_l1)

	for i, a_l0 in enumerate(weights_l1):  # for each list of weights of neurons of l1
		for j, w_ail0_ajl1 in enumerate(a_l0):  # for each weight from a_l0_i to a_l1_j
			activations_l1[i] += w_ail0_ajl1 * activations_l0[j]

	return activations_l1


def add_biases(activations: list[float], biases: list[float]):
	for i, bias in enumerate(biases):
		activations[i] += bias
	return activations


def propagate(weights_l1: list[list[float]], activations_l0: list[float], biases: vector):  # calculate next layer activations

	weighted_sum: vector = propagate_weights(weights_l1, activations_l0)
	z: vector = add_biases(weighted_sum, biases)

	l1: vector = relu(z)

	return l1
