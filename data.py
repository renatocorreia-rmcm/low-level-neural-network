from __future__ import annotations  # allows to refeer to Vector type inside Vector class definition

"""
def __str__ tells how object is represented as string,
as is requested to it when you call print(object)
default object string representation is it address
"""
""" maybe define Line type, so a matrix would be a list of lines

contra: pratically would be a vector
pro:	teoretically would be the weights of a single neuron
"""


class Vector:
	value: list[float]

	def __init__(self, value: list[float] = None, size: int = None):
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
				value = [0]*size
		self.value = value

	def append(self, data: float):
		self.value.append(data)

	def __str__(self):  # string where each line corresponds to a vector element
		text = """"""
		for data in self.value[:-1]:
			text += f"{data}\n"
		text += str(self.value[-1])
		return text

	def __len__(self):
		return len(self.value)

	def __getitem__(self, index: int):
		return self.value[index]

	def __setitem__(self, index: int, data: float):
		self.value[index] = data

	def __iter__(self):
		return iter(self.value)

	def __add__(self, other: Vector):
		out: Vector = Vector(size=len(self))
		for i, data in enumerate(self):
			out[i] = self[i]+other[i]
		return out


class Matrix:
	value: list[list[float]]

	def __init__(self, value: list[list[float]] = None):
		if value is None:
			value = []
		self.value = value

	def append(self, line: list[float]):
		self.value.append(line)

	def __getitem__(self, index: int):
		return self.value[index]

	def __str__(self):  # string where each line corresponds to a matrix line
		text = """"""
		for line in self.value[:-1]:
			text += f"{line}\n"
		text += str(self.value[-1])
		return text

	def __iter__(self):  # defines an interator to object allowing enumerate(matrix)  # the iterator created in this case its just the default iterator for list[list[float]]
		return iter(self.value)

	def __len__(self):
		return len(self.value)

	def __mul__(self, vector: Vector):
		out: Vector = Vector([0]*len(self))
		for i_line, line in enumerate(self):
			for i_coll, num in enumerate(line):
				out[i_line] += num*vector[i_coll]
		return out
