"""
implement layer object
own weights
"""

from random import randint

from data import Vector
from data import Matrix


def relu(z: Vector) -> Vector:
	out: Vector = Vector(size=len(z))
	for i, activation in enumerate(z):
		out[i] = max(0.0, activation)
	return out


class NeuralNetwork:
	# layers
	n_layers: int
	layers_sizes: list[int]
	layers: list[Vector]
	# parameters
	weights: list[Matrix]
	biases: list[Vector]

	"""
		processing
	"""

	def activation(self, i_l1: int) -> None:
		i_l0: int = i_l1-1
		l0: Vector = self.layers[i_l0]
		# calculate activation function parameters
		weights_l1: Matrix = self.weights[i_l1-1]
		biases: Vector = self.biases[i_l1-1]
		# aplly activation fuction
		self.layers[i_l1] = relu(weights_l1*l0 + biases)  # error: len(biases)<len(weighted_l1*l0)

	def process(self, inp: Vector) -> Vector:
		self.layers[0] = inp  # load input
		# execute propagation for each layer
		for i_layer in range(self.n_layers - 1):
			self.activation(i_layer+1)
		return self.layers[-1]

	"""
		analisis
	"""

	def analyse(self):
		self.print_activations()
		self.print_weights()
		self.print_biases()

	def print_activations(self):
		print("\nACTIVATIONS:\n")
		for i_layer, layer in enumerate(self.layers):
			activations: list[str] = str(layer).split('\n')
			for i_neuron, activation in enumerate(activations):
				print(f"layer_{i_layer} neuron_{i_neuron}: {activation}")
			print()

	def print_weights(self):
		print("\nWEIGHTS:\n")
		for i_layer, matrix in enumerate(self.weights):
			neurons_weights: list[str] = str(matrix).split('\n')
			for i_neuron, weights in enumerate(neurons_weights):
				print(f"layer_{i_layer+1} neuron_{i_neuron}: {weights}")
			print()

	def print_biases(self):
		print("\nBIASES:\n")
		for i_layer, layer_biases in enumerate(self.biases):
			biases: list[str] = str(layer_biases).split('\n')
			for i_neuron, bias in enumerate(biases):
				print(f"layer_{i_layer+1} neuron_{i_neuron}: {bias}")
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
		# create new matrixes wich represent (l_(i+1) weights)
		for i_layer in range(self.n_layers - 1):
			weights: Matrix = Matrix()
			size_l0 = self.layers_sizes[i_layer]  # size of output layer
			size_l1 = self.layers_sizes[i_layer + 1]  # size of input layer
			# create new lines wich represent (a_(l1, i_line) weights)
			for i_line in range(size_l1):
				line: list[float] = []
				# create new colls wich represent (a_(l1, i_line), a_(l0, i_col)) weight
				for i_coll in range(size_l0):
					weight: float = randint(-10, 10) / 10
					line.append(weight)
				weights.append(line)
			self.weights.append(weights)

		"""initialize biases"""
		# create and fill biases with random biases
		self.biases = []
		# create new biases for layer_i+1
		for i_layer in range(self.n_layers - 1):
			biases: Vector = Vector()
			layer_size = layers_sizes[i_layer + 1]
			# fill new biases with random bias
			for i_bias in range(layer_size):
				bias: float = randint(-10, 10) / 10
				biases.append(bias)
			self.biases.append(biases)
