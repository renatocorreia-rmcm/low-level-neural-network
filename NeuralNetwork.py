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

from random import randint  # parameters initialization
from random import shuffle  # data_set shuffle in each epoch

from modules.data import Vector
from modules.data import Matrix

from modules.mathematical_functions import sigmoid
from modules.mathematical_functions import sigmoid_derivative
from modules.mathematical_functions import quadratic_loss
from modules.mathematical_functions import quadratic_loss_derivative

from modules.file_functions import get_line
from modules.file_functions import set_line

"""
	CLASS
"""


class NeuralNetwork:
	"""
		fully connected neural network of given size.
		It name and parameters may be backuped automatically in a .txt,
		well as loaded from the.txt back to the object.

		Uses sigmoid as activation function
		Uses quadratic loss as cost function
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
		:param i_l1: index of layer to compute activations based in previous one (expected already calculated)
		"""
		i_l0: int = i_l1 - 1
		# calculate activation function parameters
		l0: Vector = self.layers[i_l0]
		weights_l1: Matrix = self.weights[i_l1]
		biases_l1: Vector = self.biases[i_l1]
		z_l1: Vector = weights_l1 * l0 + biases_l1
		# apply activation function
		self.layers[i_l1] = sigmoid(z_l1)

	def process_feature(self, feature: list[float]) -> list[float]:
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

	def get_answer(self) -> int:
		"""
		:return: index of brighter output activation
		"""
		i_maior: int = 0
		maior: float = self.layers[-1][i_maior]
		for i in range(1, self.layers_sizes[-1]):
			if self.layers[-1][i] > maior:
				i_maior = i
				maior = self.layers[-1][i]
		return i_maior

	"""
		learning
	"""

	# parameters
	weights_gradients: list[Matrix]
	biases_gradients: list[Vector]

	learn_step: float

	# operations

	def get_accuracy(self) -> float:
		"""
		process all trainning set
		:return: accuracy between 0 and 1
		"""
		acertos: int = 0

		for x_test, y_test in zip(self.x_test, self.y_test):
			self.process_feature(x_test)
			if self.get_answer() == y_test:
				acertos += 1

		return acertos/len(self.y_test)

	def train_epoch(self, learn_step: float = 1) -> None:
		"""
			process and backpropagate all data_points in training_set
		"""

		batch_size: int = 16
		n_batches: int = int(len(self.y_train) / batch_size)

		# random indexes order (to iterate over trainining set)
		order: list[int] = list(range(len(self.x_train)))
		shuffle(order)
		i2: int = 0  # second order index (to iterate random indexes list)

		# somatory of each batch average gradient
		for i_batch in range(n_batches):

			# reset gradient
			for i_layer in range(self.n_layers):
				self.weights_gradients[i_layer].clean()
				self.biases_gradients[i_layer].clean()

			# somatory of each data_point gradient
			for data_point in range(batch_size):

				i1: int = order[i2]  # first order index (to iterate directly over training set)
				i2 += 1

				feature: list[float]
				target: int
				feature = self.x_train[i1]
				target = self.y_train[i1]

				# forward go - compute prediction
				prediction: Vector = Vector(self.process_feature(feature))
				# reach end - compute cost
				cost: float = quadratic_loss(prediction, target)
				# backward go - calculate error signal for each layer then gradient for each parameter
				target_vec: Vector = Vector(size=self.layers_sizes[-1])
				target_vec[target] = 1
				gradients: tuple[list[Matrix], list[Vector]] = self.backpropagate(target_vec)
				weights_gradients: list[Matrix] = gradients[0]
				biases_gradients: list[Vector] = gradients[1]
				for i_layer in range(1, self.n_layers):
					self.weights_gradients[i_layer] += weights_gradients[i_layer]
					self.biases_gradients[i_layer] += biases_gradients[i_layer]

			# transform batch somatory into batch average
			for i_layer in range(1, self.n_layers):
				self.weights_gradients[i_layer] /= batch_size
				self.biases_gradients[i_layer] /= batch_size
			# subtract each partial derivative average from its corresponding parameter  # nn -= ▽C
			for i_layer in range(1, self.n_layers):
				self.weights[i_layer] -= self.weights_gradients[i_layer] * self.learn_step
				self.biases[i_layer] -= self.biases_gradients[i_layer] * self.learn_step

	def backpropagate(self, target: Vector) -> tuple[list[Matrix], list[Vector]]:
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

	def compute_gradient(self, error_signals: list[Vector]) -> tuple[list[Matrix], list[Vector]]:
		"""
		:return: 2-uple of lists of gradients (weights_gradients, biases_gradients)
		"""

		# first layer has no parameters so they start with None
		weights_gradients: list[Matrix] = [Matrix(None)]
		biases_gradients: list[Vector] = [Vector(None)]

		for i_layer in range(1, self.n_layers):
			# (▽ Wl C) = δ_l * (a_l-1)T
			weights_gradient_i = error_signals[i_layer] * (self.layers[i_layer - 1].transposed())
			weights_gradients.append(weights_gradient_i)
			# (▽ bl C) = δ_l
			biases_gradient_i = error_signals[i_layer]
			biases_gradients.append(biases_gradient_i)

		gradient: tuple[list[Matrix], list[Vector]] = (weights_gradients, biases_gradients)
		return gradient

	"""
		parameters backup
	"""

	# parameters

	file_path: str  # path to .txt from instances_backups/

	# operations

	def store(self) -> None:
		with open(self.file_path, 'w') as file:

			set_line(file, self.layers_sizes)
			set_line(file)

			for i_layer, layer_size in enumerate(self.layers_sizes[1:], 1):
				# set matrix
				for i_neuron in range(layer_size):
					set_line(file, self.weights[i_layer][i_neuron])
				set_line(file)
				# set bias
				set_line(file, self.biases[i_layer].value)
				set_line(file)

	def load(self) -> None:
		with open(self.file_path, 'r') as file:

			get_line(file)

			for i_layer, layer_size in enumerate(self.layers_sizes[1:], 1):
				# get matrix
				for i_neuron in range(layer_size):
					self.weights[i_layer][i_neuron] = get_line(file)
				# get bias
				self.biases[i_layer] = Vector(get_line(file))

	"""
		constructor
	"""

	def __init__(self, layers_sizes: list[int], backup_file_name: str, data_set: tuple[tuple[list[list[float]], list[float]], tuple[list[list[float]], list[float]]]) -> None:
		"""

		"""

		self.data_set = data_set
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.data_set

		"""backup"""
		self.file_path = f"instances_backups/{backup_file_name}.txt"

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
					weight: float = ((6 / (
								self.layers_sizes[i_layer - 1] + self.layers_sizes[i_layer])) ** 0.5) * randint(-1, 1)
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
			self.biases_gradients.append(Vector([0] * layer_size))

			# fill new biases with random bias
			for i_bias in range(layer_size):
				bias: float = (-0.1, 0.1)[randint(0, 1)]
				biases.append(bias)
			self.biases.append(biases)
