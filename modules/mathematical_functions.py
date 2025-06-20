"""
    auxiliary mathematical functions
    used in neural network implementation
"""

from modules.data import Vector
from math import e


def sigmoid(z: Vector) -> Vector:
	out: Vector
	out = (1 / (1 + (e ** (-z))))
	return out


def sigmoid_derivative(z: Vector) -> Vector:
	out: Vector = sigmoid(z) * (1 - sigmoid(z))
	return out


def quadratic_loss(prediction: Vector, target: Vector) -> float:
	cost: float = 0
	for i_activation in range(len(target)):
		cost += (prediction[i_activation] - target[i_activation])**2
	return cost


def quadratic_loss_derivative(aL: Vector, target: Vector) -> Vector:
	"""
		WITH RESPECT TO aL
	"""
	out: Vector = 2*(aL-target)
	return out
