# Section 4, Lecture 9

import numpy as np

def forward(X, W1, b1, W2, b2):
	Z = X.dot(W1) + b1
	
	#Z = 1 / (1 + np.exp(-Z))	# sigmoid
	#Z = np.tanh(Z)				# tanh
	Z[Z < 0] = 0				# relu
		
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
		
	return Y, Z

def derivative_W2(Z, pY, T):
	return Z.T.dot(pY - T)

def derivative_b2(pY, T):
	return (pY - T).sum(axis = 0)

def derivative_W1(X, W2, Z, pY, T):
	#dZ = (pY - T).dot(W2.T) * Z * (1 - Z)	# sigmoid
	#dZ = (pY - T).dot(W2.T) * (1 - Z * Z)	# tanh
	dZ = (pY - T).dot(W2.T) * (Z > 0)		# relu
	return X.T.dot(dZ)

def derivative_b1(W2, Z, pY, T):
	#dZ = (pY - T).dot(W2.T) * Z * (1 - Z)	# sigmoid
	#dZ = (pY - T).dot(W2.T) * (1 - Z * Z)	# tanh
	dZ = (pY - T).dot(W2.T) * (Z > 0)		# relu
	return dZ.sum(axis = 0)
