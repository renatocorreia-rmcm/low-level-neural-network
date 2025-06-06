def linear_transform(m: list, v: list):
	out = [0] * len(v)

	for i, dim in enumerate(m):
		for j, basis_vec in enumerate(dim):
			out[i] += basis_vec * v[j]

	return out


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
