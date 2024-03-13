import numpy as np

# Example matrix A
A = np.array([[1, 2], [3, 4], [5, 6]])

# Perform SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("U:", U)
print("S:", S)
print("Vt:", Vt)
