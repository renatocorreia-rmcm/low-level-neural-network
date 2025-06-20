# .TXT MUST HAVE AT LEAST 1 BLANC AT THE END

def float_list(numbers: list[str]):
	return [float(n) for n in numbers if n != '']


def get_line(file) -> list[float]:
	line: str = ''

	while line == '':
		line = (file.readline()[:-1])

	numbers: list[float] = float_list(line.split(" "))  # refers to curent line
	return numbers


def string_line(numbers: list[float]):
	line = ""
	for num in numbers:
		line += f"{num} "
	line = line[:-1] + "\n"

	return line


def set_line(file, numbers: list[float] = None):
	if numbers is None:
		numbers = []
	line: str
	line = string_line(numbers)
	file.write(line)

