"""
	Vector and Matrix types.
"""

from __future__ import annotations  # allows to refeer to Vector type inside Vector class definition


class Vector:
	value: list[float]

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

	def reset(self):  # set all values to 0 keeping dimention constant
		self.value = [0] * len(self)

	def append(self, data: float) -> None:
		self.value.append(data)

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

	def __getitem__(self, index: int) -> float:
		return self.value[index]

	def __setitem__(self, index: int, data: float) -> None:
		self.value[index] = data

	def __iter__(self):  # -> Iterator wich shoul be imported from Typing
		return iter(self.value)

	def __add__(self, other: any) -> Vector:
		# argument checking
		if isinstance(other, Vector):
			pass
		elif isinstance(other, (int, float)):
			other: Vector = Vector([other] * len(self))
		else:
			print("type error")
		# calculus
		out: Vector = self
		for i in range(len(self)):
			out[i] += other[i]
		return out

	def __radd__(self, other: any):
		return self.__add__(other)

	def __sub__(self, other: any) -> Vector:
		return self.__add__(-other)

	def __rsub__(self, other: any) -> Vector:
		return -self.__sub__(other)

	def __rtruediv__(self, other: float) -> Vector:
		out: Vector = Vector([other]*len(self))

		for i in range(len(self)):
			out[i] /= self[i]

		return out

	def __rmul__(self, scalar: int):
		out: Vector = self
		for i in range(len(self)):
			out[i] *= 2
		return out

	def __neg__(self):
		out: Vector = self
		for i in range(len(self)):
			out[i] *= -1
		return out

	def __rpow__(self, base: float):
		out: Vector = self
		for i in range(len(self)):
			out[i] = base ** out[i]
		return out

	def __mul__(self, other: Vector):  # Hadamard Product
		out: Vector = self
		for i in range(len(other)):
			out[i] *= other[i]
		return out


class Matrix:
	value: list[list[float]]

	def __init__(self, value: list[list[float]] = None) -> None:
		if value is None:
			value = []
		self.value = value

	def reset(self):  # set all values to 0 keeping dimensions constant
		self.value = [
						 [0] * len(self[0])
					 ] * len(self)

	def append(self, line: list[float]) -> None:
		self.value.append(line)

	def __getitem__(self, index: int) -> list[float]:
		return self.value[index]

	def __str__(self) -> str:
		"""

		:return: string where each line corresponds to a matrix line
		"""
		text = """"""
		for line in self.value[:-1]:  # REPLACE BY JOIN
			text += f"{line}\n"
		text += str(self.value[-1])
		return text

	def __iter__(
			self):  # -> Iterator wich should be imported from Typing  # defines an interator to object allowing enumerate(matrix)  # the iterator created in this case its just the default iterator for list[list[float]]
		return iter(self.value)

	def __len__(self) -> int:
		return len(self.value)

	def __mul__(self, vector: Vector) -> Vector:
		out: Vector = Vector(size=len(self))
		for i_line, line in enumerate(self):
			for i_coll, num in enumerate(line):
				out[i_line] += num * vector[i_coll]
		return out

	def __sub__(self, other: Matrix):
		out: Matrix = Matrix()
		for i_line in range(len(self)):
			out.append([])
			for i_coll in range(len(self[0])):
				out[i_line].append(self[i_line][i_coll] - other[i_line][i_coll])

		return out
