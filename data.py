"""
	Vector and Matrix types.
"""

from __future__ import annotations  # allows to refer to Vector type inside Vector class definition


class Vector:
	"""
		list of floats
		operators overloaded to linear algebra based operations.
	"""

	value: list[float]

	"""
		constructor
	"""

	# there should be a better way to define that
	def __init__(self, value: list[float] = None, size: int = None) -> None:
		"""
		Do not pass both arguments

		Default Vector: [ ]

		:param value: ready list
		:param size: size of {0} vector
		"""
		if value is None:
			if size is None:
				value = []
			else:
				value = [0] * size
		self.value = value

	"""
		structural operations
	"""

	def copy(self) -> Vector:
		return Vector(self.value.copy())

	def __list__(self) -> list[float]:
		return self.value

	def __str__(self) -> str:
		"""

		:return: string where each line corresponds to a vector element
		"""
		text = """"""
		for data in self.value[:-1]:
			text += f"{data}\n"
		text += str(self.value[-1])
		return text

	def __len__(self) -> int:
		return len(self.value)

	def __iter__(self):  # -> Iterator which should be imported from Typing
		return iter(self.value)

	def transposed(self):  # returns (1 x n) matrix
		out: Matrix
		out = Matrix([self.value])
		return out

	"""
		data operations
	"""

	def clean(self):  # set all values to 0 keeping dimensions sizes constant
		self.value = [0] * len(self)

	def append(self, data: float) -> None:
		self.value.append(data)

	def __getitem__(self, index: int) -> float:
		return self.value[index]

	def __setitem__(self, index: int, data: float) -> None:
		self.value[index] = data

	"""
		mathematical operations
	"""

	def __neg__(self):
		# element wise
		out: Vector = self.copy()
		for i in range(len(out)):
			out[i] *= -1
		return out

	def __add__(self, other: any) -> Vector:
		# Vector + Vector
		if isinstance(other, Vector):
			out: Vector = other.copy()
			for i in range(len(self)):
				out[i] += other[i]
			return out
		# Vector + scalar  # element wise
		elif isinstance(other, (int, float)):
			out: Vector = self.copy()
			for i in range(len(self)):
				out[i] += other
			return out
		# argument type error
		return None

	def __radd__(self, other: any) -> Vector:
		# commutative
		return self.__add__(other)

	def __sub__(self, other: Vector) -> Vector:
		return self.__add__(-other)

	def __rsub__(self, other: any) -> Vector:
		return -(self.__sub__(other))

	def __truediv__(self, other: float):
		return self.__mul__(1/other)

	def __rtruediv__(self, other: float) -> Vector:
		# element wise
		out: Vector = Vector([other]*len(self))
		for i in range(len(self)):
			out[i] /= self[i]
		return out

	def __mul__(self, other: any) -> any:
		# hadamard Product
		if isinstance(other, Vector):
			out: Vector = self.copy()
			for i in range(len(out)):
				out[i] *= other[i]
			return out
		# scalar product
		elif isinstance(other, (int, float)):
			out: Vector = self.copy()
			for i in range(len(out)):
				out[i] *= other
			return out
		# column_vector x line_matrix
		elif isinstance(other, Matrix):
			collumn_matrix: Matrix
			collumn_matrix = Matrix([[self[i]] for i in range(len(self))])
			return collumn_matrix*other

		return None

	def __rmul__(self, other: any):
		# commutative
		return self.__mul__(other)

	def __rpow__(self, base: float):
		# element wise
		out: Vector = self.copy()
		for i in range(len(self)):
			out[i] = base ** out[i]
		return out


class Matrix:
	"""
		list of lists of floats
		operators overloaded to linear algebra based operations.
	"""

	value: list[list[float]]


	"""
		constructor
	"""

	def __init__(self, value: list[list[float]] = None) -> None:  # todo: implement [[0] * size_coll for _ in range(size_line)] directly into constructor
		if value is None:
			value = []
		self.value = value.copy()


	"""
		structural operations
	"""

	def copy(self) -> Matrix:
		return Matrix(self.value.copy())

	def clean(self) -> None:  # set all values to 0 keeping dimensions constant
		for i in range(len(self)):
			for j in range(len(self[0])):
				self[i][j] = 0

	def __str__(self) -> str:
		"""

		:return: string where each line corresponds to a matrix line
		"""
		text = """"""
		for line in self.value[:-1]:
			text += f"{line}\n"
		text += str(self.value[-1])
		return text

	def __iter__(self):  # -> Iterator which should be imported from Typing  # defines an interator to object allowing enumerate(matrix)  # the iterator created in this case its just the default iterator for list[list[float]]
		return iter(self.value)

	def __len__(self) -> int:
		return len(self.value)

	def transpose(self) -> Matrix:
		"""
		return new object transposed. don't modify caller object
		"""
		out = Matrix([[] for _ in range(len(self[0]))])  # create out with c lines, where c is the number of collumns of self
		for line in self:
			for i_coll, coll in enumerate(line):
				out[i_coll].append(coll)

		return out

	"""
		data operations
	"""

	def append(self, line: list[float]) -> None:
		self.value.append(line)

	def __getitem__(self, index: int) -> list[float]:
		return self.value[index]

	"""
		mathematical operations
	"""

	def __mul__(self, other: any) -> any:
		if isinstance(other, Vector):  # matrix x vector
			vec: Vector = other.copy()
			out: Vector = Vector(size=len(self))  # (n x m) x (m x 1)  ->  (n x 1)
			for i_line, line in enumerate(self):
				for i_coll, num in enumerate(line):
					out[i_line] += num * vec[i_coll]

			return out
		elif isinstance(other, Matrix):  # matrix x matrix
			out: Matrix = Matrix([[0]*len(other[0]) for _ in range(len(self))])
			for i_line, line in enumerate(self):
				for i_coll_other in range(len(other[0])):
					for i_line_other in range(len(other)):
						out[i_line][i_coll_other] += self[i_line][i_line_other]*other[i_line_other][i_coll_other]

			return out
		elif isinstance(other, float):  # matrix * scalar
			out: Matrix = self.copy()
			for i_line in range(len(self)):
				for i_coll in range(len(self[0])):
					out[i_line][i_coll] *= other

			return out

	def __truediv__(self, other: float) -> Matrix:
		return self.__mul__(1/other)

	def __sub__(self, other: Matrix) -> Matrix:
		out: Matrix = self.copy()
		for i_line in range(len(self)):
			for i_coll in range(len(self[0])):
				out[i_line][i_coll] -= other[i_line][i_coll]

		return out

	def __add__(self, other: Matrix) -> Matrix:
		out: Matrix = self.copy()
		for i_line in range(len(other)):
			for i_coll in range(len(other)):
				out[i_line][i_coll] += other[i_line][i_coll]

		return out
