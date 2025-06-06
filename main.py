import numpy
from random import randint
from propagate import propagate


"""
	SETUP NEURAL NETWORK
"""

layers_sizes = [5, 3, 2]

n_layers = len(layers_sizes)


"""
	initialize layers
"""

layers = []
for layer_size in layers_sizes:
	layers.append([0] * layer_size)


"""
	initialize weights
"""

weights = [  # list of matrices for layer_i

]

for i in range(0, n_layers-1):  # fill weights

	weights.append([])  # create new matrix (layer i weights)

	size_l0 = layers_sizes[i]  # out layer
	size_l1 = layers_sizes[i+1]  # in layer

	for line in range(size_l1):
		weights[i].append([])  # create new line (a(l1, line) weights)

		for coll in range(size_l0):
			random: float = randint(0, 10)/10
			weights[i][line].append(random)  # create new coll (random weight)


"""
	print weights
"""
for i, matrix in enumerate(weights):
	print(f"layer {i+1} weights")
	for line in matrix:
		print(line)
	print()


"""
	initialize biases
"""
biases = [  # list of vectors for layer_i

]
