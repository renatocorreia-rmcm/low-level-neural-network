type vector = list[float]
type matrix = list[list[float]]

# propagate activations from input to output

from propagate import *


def process(inp: list[float], weights: list[list[list[float]]], biases: list[vector], n_layers: int):
	current_layer = inp
	for l in range(n_layers-1):
		current_layer = propagate(weights[l], current_layer, biases[l])

	return current_layer
