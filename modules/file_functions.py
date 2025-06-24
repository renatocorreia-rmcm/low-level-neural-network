# .TXT MUST HAVE AT LEAST 1 BLANC AT THE END

def float_list(numbers: list[str]) -> list[float]:
	"""
	:param numbers: list of floats stored as strings
	:return: converted list of floats
	"""
	return [float(n) for n in numbers if n != '']


def get_line(file) -> list[float]:
	"""
	get file line as if it was a stack
	ignore blanks

	:param file:  file object opened in r mode
	:return: file line as a list of float
	"""
	line: str = ''

	while line == '':
		line = (file.readline()[:-1])

	numbers: list[float] = float_list(line.split(" "))  # refers to curent line
	return numbers


def string_line(numbers: list[float]) -> str:
	"""
	:param numbers: list of float
	:return: numbers list converted to a single string whitespaced
	"""
	line: str = ""
	for num in numbers:
		line += f"{num} "
	line = line[:-1] + "\n"

	return line


def set_line(file, numbers: list[float] = None) -> None:
	"""
	:param file: file object opened in w mode
	:param numbers: the line to append as file was a stack
	"""
	if numbers is None:
		numbers = []
	line: str
	line = string_line(numbers)
	file.write(line)

