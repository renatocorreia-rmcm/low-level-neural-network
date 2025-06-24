"""
    auxiliary mathematical functions
    used in neural network implementation
"""

from modules.data import Vector
from math import e


def sigmoid(z: Vector) -> Vector:
	"""
	logistic function
	applied element-wise in a Vector
	"""
	out: Vector = Vector(size=len(z))
	for i in range(len(out)):
		try:
			out[i] = 1 / (1 + (e ** -z[i]))
		except OverflowError:
			out[i] = 0
	return out


def sigmoid_derivative(z: Vector) -> Vector:
	"""
	WITH RESPECT TO z

	derivative of logistic function
	applied element-wise in a vector
	"""
	out: Vector = sigmoid(z) * (1 - sigmoid(z))
	return out


def quadratic_loss(prediction: Vector, target: int) -> float:
	"""
	somatory of the square of the difference between prediction and target
	element wise in a Vector

	:return: the cost
	"""
	cost: float = 0
	for i_activation in range(10):
		if i_activation == target:
			cost += (prediction[i_activation]-1)**2
		else:
			cost += prediction[i_activation]**2
	return cost


def quadratic_loss_derivative(aL: Vector, target: Vector) -> Vector:
	"""
	WITH RESPECT TO aL

	applied element-wise in a Vector
	"""
	out: Vector = 2*(aL-target)
	return out
