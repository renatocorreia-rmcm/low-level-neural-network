type vector = list[float]
type matrix = list[list[float]]

import numpy
from random import randint
from propagate import propagate_weights
from activation import relu
from process import process

"""
	SETUP NEURAL NETWORK
"""

layers_sizes = [3, 2, 1]  # layer_0 = input

inp = [
	30,
	20,
	10,
]


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

for i in range(n_layers - 1):  # fill weights

	weights.append([])  # create new matrix (layer i+1 weights)

	size_l0 = layers_sizes[i]  # out layer
	size_l1 = layers_sizes[i + 1]  # in layer

	for line in range(size_l1):
		weights[i].append([])  # create new line (a(l1, line) weights)

		for coll in range(size_l0):
			random: float = randint(-10, 10) / 10
			weights[i][line].append(random)  # create new coll (random weight)

"""
	print weights
"""
for i, matrix in enumerate(weights):
	print(f"layer {i + 1} weights")
	for line in matrix:
		print(line)
	print()

"""
	initialize biases
"""

biases = [  # list of vectors for layer_i

]

for i in range(n_layers - 1):
	biases.append([])  # create bias for layer_i+1

	layer_size = layers_sizes[i + 1]

	for b in range(layer_size):
		biases[i].append(randint(-10, 10) / 10)

"""
	print biases
"""

for i, vector in enumerate(biases):
	print(f"layer {i + 1} biases")
	for bias in vector:
		print(bias)
	print()


layers[0] = inp


print(process(layers[0], weights, biases, n_layers))
