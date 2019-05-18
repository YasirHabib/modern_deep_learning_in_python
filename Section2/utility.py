# Section 2, Lecture 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_transformed_data():
	
	df = pd.read_csv("train.csv")
	data = df.values.astype(np.float32)
	np.random.shuffle(data)
	X = data[:, 1:]
	Y = data[:, 0]
	
	# split train and test
	X_train = X[:-1000]
	Y_train = Y[:-1000]
	X_test = X[-1000:]
	Y_test = Y[-1000:]
	
	# center the data
	mu = X_train.mean(axis=0)
	X_train = X_train - mu
	X_test  = X_test - mu
	
	# transform the data
	pca = PCA()
	Z_train = pca.fit_transform(X_train)
	Z_test = pca.transform(X_test)
	
	# take first 300 cols of Z
	Z_train = Z_train[:, :300]
	Z_test = Z_test[:, :300]
	
	# normalize Z
	Z_train = (Z_train - Z_train.mean(axis = 0)) / Z_train.std(axis = 0)
	Z_test = (Z_test - Z_test.mean(axis = 0)) / Z_test.std(axis = 0)
	
	return Z_train, Z_test, Y_train, Y_test
	
def y2indicator(y):
	N = len(y)
	K = len(set(y))
	y = y.astype(np.int32)
	T = np.zeros((N, K))
	for x in range(N):
		T[x, y[x]] = 1
	return T

def gradW(X, pY, T):
	return X.T.dot(pY - T)
	
def gradb(pY, T):
	return (pY - T).sum(axis=0)

def forward(X, W, b):
	# softmax
	A = X.dot(W) + b
	expA = np.exp(A)
	return expA / expA.sum(axis = 1, keepdims = True)

def predict(pY):
	return np.argmax(pY, axis = 1)
	
def error_rate(pY, Y):
	prediction = predict(pY)
	return np.mean(prediction != Y)

def cost(pY, T):
	return -(T * np.log(pY)).sum()

	

if __name__ == '__main__':
    # benchmark_pca()
    get_transformed_data()
	