type vector = list[float]
type matrix = list[list[float]]

import math


def sigmoid(z: list[float]):
	for i, activation in enumerate(z):
		z[i] = 1 / (1 + (math.e) ** (-activation))


def relu(z: list[float]):
	for i, activation in enumerate(z):
		z[i] = max(0, activation)
	return z
