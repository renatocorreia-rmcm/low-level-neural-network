"""
    auxiliary mathematical functions
    used in neural network implementation
"""

from data import Vector
from math import e

def sigmoid(z: Vector) -> Vector:
	out: Vector
	out = (1 / (1 + (e ** (-z))))
	return out


def sigmoid_derivative(z: Vector) -> Vector:
	out: Vector = sigmoid(z) * (1 - sigmoid(z))
	return out


def quadratic_loss(target: Vector, prediction: Vector) -> float:
	cost: float = 0
	for i_activation in range(len(target)):
		cost += (prediction[i_activation] - target[i_activation])
	return cost
