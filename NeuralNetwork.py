"""
	implementation of the NeuralNetwork Class
"""

""" CONVEÇÕES DESTA IMPLEMENTAÇÃO

Conveção da indexação dos parâmetros da rede neural

	pesos(i) são conexões que saem de layer(i-1) para layer(i)  
	logo, weights[i] é usado com layer[i-1] para gerar layer[i]
	primeiro elemento de self.weights: None

	biases(i) são biases que entram na layer(i)
	primeiro elemente de self.biases: None 

 
Convenção de nomeação de parâmetros referentes a camadas

	os parâmetros l0 e l1 não representam necessariamente self.layers[0] e self.layers[1],
	mas sim self.layers[i] e self.layers[i+1] respectivamente,
	para qualquer i que tenha sido passado como argumento
	
	ou seja, l0 e l1 são usados para se referir a camadas adjacentes,
	que não necessariamente são a 0 e a 1
"""

"""
	external dependencies
"""

from random import randint

from data import Vector
from data import Matrix

from mathematical_functions import sigmoid
from mathematical_functions import sigmoid_derivative
from mathematical_functions import quadratic_loss
from mathematical_functions import quadratic_loss_derivative

"""
	CLASS
"""


class NeuralNetwork:
	"""
		fully connected neural network
		of given size.
		It name and parameters may be backuped automatically in a .txt,
		well as loaded from the.txt back to the object.
	"""

	# layers
	n_layers: int
	layers_sizes: list[int]
	layers: list[Vector]  # store activations
	# parameters
	weights: list[Matrix]
	biases: list[Vector]

	"""
		processing
	"""

	def activate(self, i_l1: int) -> None:
		"""
		:param i_l1: index of layer to compute activations
		"""
		i_l0: int = i_l1 - 1
		# calculate activation function parameters
		l0: Vector = self.layers[i_l0]
		weights_l1: Matrix = self.weights[i_l1]
		biases_l1: Vector = self.biases[i_l1]
		z_l1: Vector = weights_l1 * l0 + biases_l1
		# apply activation function
		self.layers[i_l1] = sigmoid(z_l1)

	def process(self, feature: list[float]) -> list[float]:
		"""
		:return: prediction
		"""
		feature: Vector = Vector(feature)
		# load input
		self.layers[0] = feature
		# execute propagation for each layer
		for i_layer in range(1, self.n_layers):
			self.activate(i_layer)
		return list(self.layers[-1])

	"""
		learning
	"""

	# parameters
	weights_gradients: list[Matrix]
	biases_gradients: list[Vector]

	# operations

	def learn_batch(self, batch: any) -> None:  # todo: keeping batch abstract until we actually have it
		""" for data_point in batch:
				process data_point 	- forward go
				calculate cost 		- reach end
				backpropagate 		- backward go
			nn -= ▽C
		"""

		# reset gradient
		for i_layer in range(self.n_layers):
			self.weights_gradients[i_layer].clean()
			self.biases_gradients[i_layer].clean()

		# somatory of each data_point gradient
		for data_point in batch:

			feature: list[float]
			target: Vector
			feature = data_point[0]
			target = Vector(data_point[1])

			# forward go - calculate prediction
			prediction: Vector = Vector(self.process(feature))
			# reach end - calculate cost
			cost: float = quadratic_loss(prediction, target)
			# backward go - calculate error signal for each layer then gradient for each parameter
			gradients: tuple[list[Matrix], list[Vector]] = self.backpropagate(cost, target)
			weights_gradients: list[Matrix] = gradients[0]
			biases_gradients: list[Vector] = gradients[1]
			for i_layer in range(1, self.n_layers):  # todo: fix: weights_gradients is an list of None here.
				self.weights_gradients[i_layer] += weights_gradients[i_layer]
				self.biases_gradients[i_layer] += biases_gradients[i_layer]

		# transform batch somatory into batch average
		for i_layer in range(self.n_layers):
			self.weights_gradients[i_layer] /= len(batch)
			self.biases_gradients[i_layer] /= len(batch)
		# subtract each partial derivative average from its corresponding parameter  # nn -= ▽C
		for i_layer in range(1, self.n_layers):
			self.weights[i_layer] -= self.weights_gradients[i_layer]
			self.biases[i_layer] -= self.biases_gradients[i_layer]

	def backpropagate(self, cost: float, target: Vector) -> tuple[list[Matrix], list[Vector]]:
		# layers partial derivatives
		error_signals: list[Vector] = self.compute_error_signals(target)
		# gradient
		return self.compute_gradient(error_signals)

	def compute_error_signals(self, target: Vector) -> list[Vector]:
		error_signals: list[Vector] = []  # will be used as a stack, since were iterating from L to 0

		""" L """
		# δ_L =  f'(L) ⊙ ( ▽ aL C)
		error_signal_L = sigmoid_derivative(self.layers[-1]) * quadratic_loss_derivative(self.layers[-1], target)
		error_signals = [error_signal_L] + error_signals

		""" l_i """
		# δ_l = (f'(l))  ⊙  ( (Wl+1)T ⋅ δ_l+1 )
		for i_layer in range(self.n_layers - 2, 0, -1):
			error_signal_i: Vector
			next_error_signal: Vector = error_signals[0]

			error_signal_i = sigmoid_derivative(self.layers[i_layer]) * (
						self.weights[i_layer + 1].transpose() * next_error_signal)

			error_signals = [error_signal_i] + error_signals

		""" l_0 """
		error_signals = [None] + error_signals

		return error_signals

	def compute_gradient(self, error_signals: list[Vector]) -> tuple[list[Matrix],list[Vector]]:
		"""
		:return: list of gradient pairs (weights_li, biases_li)
		"""

		weights_gradients: list[Matrix] = [Matrix(None)]
		biases_gradients: list[Vector] = [Vector(None)]

		for i_layer in range(1, self.n_layers):
			# (▽ Wl C) = δ_l * (a_l-1)T
			weights_gradient_i = error_signals[i_layer]*(self.layers[i_layer-1].transposed())
			weights_gradients.append(weights_gradient_i)
			# (▽ bl C) = δ_l
			biases_gradient_i = error_signals[i_layer]
			biases_gradients.append(biases_gradient_i)

		gradient = (weights_gradients, biases_gradients)
		return gradient

	"""
		analysis
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

	"""
		constructor
	"""

	def __init__(self, layers_sizes: list[int]):
		"""

		:param layers_sizes:
		"""

		"""gradient l0"""  # others are added above together with the random initial parameters
		self.weights_gradients = [Matrix(None)]
		self.biases_gradients = [Vector(None)]

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
		self.weights = [None]  # first layer has no weights
		# create new matrixes wich represent (l_i weights)
		for i_layer in range(1, self.n_layers):
			weights_l: Matrix = Matrix()
			size_l0 = self.layers_sizes[i_layer - 1]  # size of output layer
			size_l1 = self.layers_sizes[i_layer]  # size of input layer

			"""gradient"""
			self.weights_gradients.append(Matrix([[0] * size_l0 for _ in range(size_l1)]))

			# create new lines wich represent (a_(l1, i_line) weights)
			for i_line in range(size_l1):
				line: list[float] = []
				# create new colls wich represent (a_(l1, i_line), a_(l0, i_col)) weight
				for i_coll in range(size_l0):
					weight: float = randint(-10, 10) / 10
					line.append(weight)
				weights_l.append(line)
			self.weights.append(weights_l)

		"""initialize biases"""
		# create and fill biases with random biases
		self.biases = [None]  # layer[0] (input) has no bias
		# create new biases for layer_i+1
		for i_layer in range(1, self.n_layers):
			biases: Vector = Vector()
			layer_size = layers_sizes[i_layer]

			"""gradient"""
			self.biases_gradients.append(Vector([0]*layer_size))

			# fill new biases with random bias
			for i_bias in range(layer_size):
				bias: float = randint(-10, 10) / 10
				biases.append(bias)
			self.biases.append(biases)
