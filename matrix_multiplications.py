class Matrix:  # extract what actually is used in muls
	value: list[list[float]]

	def __init__(self, value: list[list[float]] = None):
		if value is None:
			value = []
		self.value = value

	def append(self, line: list[float]):
		self.value.append(line)

	def __getitem__(self, index: int):
		return self.value[index]

	def __str__(self):  # string where each line corresponds to a matrix line
		text = """"""
		for line in self.value[:-1]:
			text += f"{line}\n"
		text += str(self.value[-1])
		return text

	def __iter__(self):  # defines an interator to object allowing enumerate(matrix)  # the iterator created in this case its just the default iterator for list[list[float]]
		return iter(self.value)


""" adapt
	def __mul__(self, vector: Vector):
		out: Vector = Vector([0]*len(vector))
		for i_line, line in enumerate(self):
			for i_coll, num in enumerate(line):
				out[i_line] += num*vector[i_coll]
		return out
"""


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

matrix = [  # shear
	[1, 1],
	[0, 1],
]

vec1 = [  # î
	1,
	0
]
vec2 = [  # j
	0,
	1
]
vec3 = [
	1,
	1
]

print(linear_transform(matrix, vec1))
print(linear_transform(matrix, vec2))
print(linear_transform(matrix, vec3))

weights_l1 = [
	[1, 2],
	[3, 4],
	[5, 6]
]

activations_l0 = [
	7,
	8
]

print(propagate_weights(weights_l1, activations_l0))

weights_l1 = [
	[1, 2, 3],
	[4, 5, 6]
]
activations_l0 = [
	7,
	8,
	9
]

print(propagate_weights(weights_l1, activations_l0))
