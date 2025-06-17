"""
implement layer object
better modularization of parameters
"""
import math


"""	Conveção da indexação dos parâmetros da rede neural

pesos(i) são conexões que saem de layer(i) para layer(i+1)  
	logo, weights[i] é usado com layer[i] para gerar layer[i+1]
	ultimo elemento de self.weights: None

biases(i) são biases que entram na layer(i)
	primeiro elemente de self.biases: NONE

# WEIGHTS[i]  PERTENCEM A LAYER[i] -> geram layer[i+1]  # i: [0, n-1[
# BIASES[i]   PERTENCEM A LAYER[i] -> geram layer[i]    # i  ]0, n-1]
"""

""" Convenção de nomeação de parâmetros referentes a camadas

l0 e l1 não representam necessariamente self.layers[0] e self.layers[1],
	mas sim self.layers[i] e self.layers[i+1] respectivamente,
	para qualquer self.layers[i] que esteja em operaração
"""

from random import randint

from data import Vector
from data import Matrix

"""
	functions
"""


def relu(z: Vector) -> Vector:
	out: Vector = Vector(size=len(z))
	for i, activation in enumerate(z):
		out[i] = max(0.0, activation)
	return out


def sigmoid(z: Vector) -> Vector:
	out: Vector
	out = (1 / (1 + (math.e ** (-z))))
	return out


def sigmoid_derivative(z: Vector) -> Vector:
	out: Vector = sigmoid(z) * (1 - sigmoid(z))
	return out


def quadratic_loss(target: Vector, prediction: Vector) -> float:
	cost: float = 0
	for i_activation in range(len(target)):
		cost += (prediction[i_activation] - target[i_activation])
	return cost


"""
	CLASS
"""


class NeuralNetwork:
	# layers
	n_layers: int
	layers_sizes: list[int]
	layers: list[Vector]  # hold activations
	# parameters
	weights: list[Matrix]
	biases: list[Vector]
	# z
	z: list[Vector]  # stored for computing partial derivative in backpropagation

	"""
		processing
	"""

	def activate(self, i_l1: int) -> None:
		i_l0: int = i_l1 - 1
		# calculate activation function parameters
		l0: Vector = self.layers[i_l0]
		weights_l0: Matrix = self.weights[i_l0]
		biases_l1: Vector = self.biases[i_l1]
		# z
		z: Vector = weights_l0 * l0 + biases_l1
		self.z[i_l1] = z  # store to access in backpropagation
		# apply activation function
		self.layers[i_l1] = relu(z)

	def process(self, feed: list[int]) -> Vector:
		feed: Vector = Vector(feed)
		# load input
		self.layers[0] = feed
		# execute propagation for each layer
		for i_layer in range(1, self.n_layers):
			self.activate(i_layer)
		return self.layers[-1]

	"""
		learning
	"""

	"""parameters"""
	# gradients - sum over all data points and take average later  # could it be permanently = {0} if all changes were done in a learn_batch call scope?
	weights_gradients: list[Matrix]
	biases_gradients: list[Vector]
	# partial derivatives - reset after each data point
	activations_partial_derivatives: list[Vector]
	z_partial_derivatives: list[Vector]

	"""operations"""

	def reset_gradient(self):
		for i_layer in range(
				self.n_layers):  # for each layer, reset his Vectors activations_gradient, biases_gradient and his Matrix weights_gradient
			self.weights_gradients[i_layer].reset()
			self.biases_gradients[i_layer].reset()

	def learn_batch(self, batch):

		for data_point in batch:
			"""cost"""
			feature, target = data_point.split()
			self.process(feature)  # calculate prediction
			cost: float = quadratic_loss(target, self.layers[-1])

			"""vector: partial derivatives of cost in relation to each activation"""
			# ∂(C0/aL)
			aL: Vector = self.layers[-1]
			vec: Vector = 2 * (aL - target)
			"""vector: partial derivatives of activations in relation to each z"""
			# ∂(aL/z)
			zL: Vector = self.z[-1]
			vec: Vector = sigmoid_derivative(zL)
			"""list[matrix]: partial derivative of zL in relation to WL"""
			# zL is a vec, while WL is a matrix. so we get a list, where each element is a matrix related to a element of ZL
			# ∂(zL/WL)
			al: Vector = self.layers[-2]  # a_l-1
			vec: Vector

	def backpropagate(self, cost: Vector, target: Vector):  # compute gradient
		return

	"""
		analisis
	"""

	def analyse(self) -> None:
		self.print_activations()
		self.print_weights()
		self.print_biases()

	def print_activations(self) -> None:
		print("\nACTIVATIONS:\n")
		for i_layer, layer in enumerate(self.layers):
			activations: list[str] = str(layer).split('\n')
			for i_neuron, activation in enumerate(activations):
				print(f"layer_{i_layer} neuron_{i_neuron}: {activation}")
			print()

	def print_weights(self) -> None:
		print("\nWEIGHTS:\n")
		for i_layer, matrix in enumerate(self.weights):
			neurons_weights: list[str] = str(matrix).split('\n')
			for i_neuron, weights in enumerate(neurons_weights):
				print(f"layer_{i_layer} neuron_{i_neuron}: {weights}")
			print()

	def print_biases(self) -> None:
		print("\nBIASES:\n")
		for i_layer, layer_biases in enumerate(self.biases):
			biases: list[str] = str(layer_biases).split('\n')
			for i_neuron, bias in enumerate(biases):
				print(f"layer_{i_layer} neuron_{i_neuron}: {bias}")
			print()

	def print_output(self) -> None:
		print("\nOUTPUT:\n")
		activations: list[str] = str(self.layers[-1]).split('\n')
		for activation in activations:
			print(activation)
		print()

	"""
		constructor
	"""

	def __init__(self, layers_sizes: list[int]):

		"""initialize layers"""
		self.layers_sizes = layers_sizes
		self.n_layers = len(layers_sizes)
		# create and fill layers with 0 activations
		self.layers = []
		# create new layers wich represent (layer_i activations)
		for layer_size in layers_sizes:
			layer: Vector = Vector(size=layer_size)
			self.layers.append(layer)

		"""initialize weights"""
		# create and fill weights with random weights
		self.weights = []
		# create new matrixes wich represent (l_i weights)
		for i_layer in range(self.n_layers - 1):  # last layer has no weights
			weights: Matrix = Matrix()
			size_l0 = self.layers_sizes[i_layer]  # size of output layer
			size_l1 = self.layers_sizes[i_layer + 1]  # size of input layer
			# create new lines wich represent (a_(l0, i_line) weights)
			for i_line in range(size_l1):
				line: list[float] = []
				# create new colls wich represent (a_(l1, i_line), a_(l0, i_col)) weight
				for i_coll in range(size_l0):
					weight: float = randint(-10, 10) / 10
					line.append(weight)
				weights.append(line)
			self.weights.append(weights)
		self.weights.append(None)  # last layer has no weights

		"""initialize biases"""
		# create and fill biases with random biases
		self.biases = [None]  # layer[0] (input) has no bias
		# create new biases for layer_i+1
		for i_layer in range(1, self.n_layers):
			biases: Vector = Vector()
			layer_size = layers_sizes[i_layer]
			# fill new biases with random bias
			for i_bias in range(layer_size):
				bias: float = randint(-10, 10) / 10
				biases.append(bias)
			self.biases.append(biases)

		"""initialize z"""
		# fill z with void vectors to overwrite it after
		for size in self.layers_sizes:
			self.z.append(Vector(size=size))
