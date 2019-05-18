# Section 3, Lecture 6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from utility import get_transformed_data, y2indicator, gradW, gradb, forward, cost, error_rate

def main():
	Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
	print("Performing logistic regression...")
	
	N, D = Xtrain.shape
	K = len(set(Ytrain))
	
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)
	
	# 1. full
	W = np.random.randn(D, K) / np.sqrt(D + K)
	b = np.zeros(K)
	
	costs = []
	lr = 0.0001
	reg = 0.01
	epochs = 50
	t0 = datetime.now()

	for t in range(epochs):
		pY = forward(Xtrain, W, b)
		
		W -= lr * (gradW(Xtrain, pY, Ytrain_ind) + reg*W)
		b -= lr * (gradb(pY, Ytrain_ind) + reg*b)

		pY_test = forward(Xtest, W, b)
		c = cost(pY_test, Ytest_ind)
		costs.append(c)

		if t % 1 == 0:
			e = error_rate(pY_test, Ytest)
			
			if t % 10 == 0:
				print("Cost at iteration %d: %.6f" % (t, c))
				print("Error rate:", e)
			
	print("Elapsted time for full GD:", datetime.now() - t0)
	print("\n")

	# 2. stochastic
	W = np.random.randn(D, K) / np.sqrt(D + K)
	b = np.zeros(K)
	
	costs_stochastic = []
	lr = 0.0001
	reg = 0.01
	epochs = 50
	t0 = datetime.now()

	for t in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for n in range(min(N, 500)): # shortcut so it won't take so long...
			x = tmpX[n, :].reshape(1, D)
			y = tmpY[n, :].reshape(1, K)
		
			pY = forward(x, W, b)
		
			W -= lr * (gradW(x, pY, y) + reg*W)
			b -= lr * (gradb(pY, y) + reg*b)

			pY_test = forward(Xtest, W, b)
			c = cost(pY_test, Ytest_ind)
			costs_stochastic.append(c)

		if t % 1 == 0:
			e = error_rate(pY_test, Ytest)
			
			if t % 10 == 0:
				print("Cost at iteration %d: %.6f" % (t, c))
				print("Error rate:", e)
			
	print("Elapsted time for SGD:", datetime.now() - t0)
	print("\n")
	
	# 3. batch
	W = np.random.randn(D, K) / np.sqrt(D + K)
	b = np.zeros(K)
	
	costs_batch = []
	lr = 0.0001
	reg = 0.01
	batch_sz = 500
	n_batches = N // batch_sz
	epochs = 50
	t0 = datetime.now()

	for t in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			pY = forward(x, W, b)
		
			W -= lr * (gradW(x, pY, y) + reg*W)
			b -= lr * (gradb(pY, y) + reg*b)

			pY_test = forward(Xtest, W, b)
			c = cost(pY_test, Ytest_ind)
			costs_batch.append(c)

		if t % 1 == 0:
			e = error_rate(pY_test, Ytest)
			
			if t % 10 == 0:
				print("Cost at iteration %d: %.6f" % (t, c))
				print("Error rate:", e)
			
	print("Elapsted time for batch GD:", datetime.now() - t0)
	
	x1 = np.linspace(0, 1, len(costs))
	plt.plot(x1, costs, label="full")
	x2 = np.linspace(0, 1, len(costs_stochastic))
	plt.plot(x2, costs_stochastic, label="stochastic")
	x3 = np.linspace(0, 1, len(costs_batch))
	plt.plot(x3, costs_batch, label="batch")
	plt.legend()
	plt.show()
	
if __name__ == '__main__':
	main()