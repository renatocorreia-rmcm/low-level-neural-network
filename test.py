from propagate import linear_transform

matrix = [  # shear
	[1, 1],
	[0, 1],
]

vec1 = [  # î
	1,
	0
]
vec2 = [  # j
	0,
	1
]
vec3 = [
	1,
	1
]

print(linear_transform(matrix, vec1))
print(linear_transform(matrix, vec2))
print(linear_transform(matrix, vec3))


from propagate import propagate

weights_l1 = [
	[1, 2],
	[3, 4],
	[5, 6]
]

activations_l0 = [
	7,
	8
]

print(propagate(weights_l1, activations_l0))

weights_l1 = [
	[1, 2, 3],
	[4, 5, 6]
]
activations_l0 = [
	7,
	8,
	9
]

print(propagate(weights_l1, activations_l0))
