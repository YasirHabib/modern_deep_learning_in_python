# Section 4, Lecture 11

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utility import get_transformed_data, y2indicator, cost, error_rate
from mlp import forward, derivative_W2, derivative_b2, derivative_W1, derivative_b1

def main():

	Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
	
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)
	
	N, D = Xtrain.shape
	M = 300
	K = len(set(Ytrain))
		
	W1 = np.random.randn(D, M) / np.sqrt(D)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.zeros(K)
	
	# save initial weights
	W1_0 = W1.copy()
	b1_0 = b1.copy()
	W2_0 = W2.copy()
	b2_0 = b2.copy()
	
	lr = 0.00004
	reg = 0.01
	batch_sz = 500
	n_batches = N // batch_sz
	epochs = 20
	
	# 1. batch
	costs_batch = []
	for t in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			pY, Z = forward(x, W1, b1, W2, b2)
		
			W2 -= lr * (derivative_W2(Z, pY, y) + reg*W2)
			b2 -= lr * (derivative_b2(pY, y) + reg*b2)
			W1 -= lr * (derivative_W1(x, W2, Z, pY, y) + reg*W1)
			b1 -= lr * (derivative_b1(W2, Z, pY, y) + reg*b1)
			
			if j % 10 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_batch.append(c)
				print("Cost at iteration t=%d, j=%d: %.6f" % (t, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
	print("\n")
	
	# 2. RMSprop
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	
	cache_W2 = 1
	cache_b2 = 1
	cache_W1 = 1
	cache_b1 = 1
	decay_rate = 0.999
	eps = 1e-10
	lr0 = 0.001
	
	costs_RMS = []
	for t in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			pY, Z = forward(x, W1, b1, W2, b2)
		
			gW2 = (derivative_W2(Z, pY, y) + reg*W2)
			cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
			W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)
			
			gb2 = (derivative_b2(pY, y) + reg*b2)
			cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
			b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)
			
			gW1 = (derivative_W1(x, W2, Z, pY, y) + reg*W1)
			cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
			W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)
			
			gb1 = (derivative_b1(W2, Z, pY, y) + reg*b1)
			cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
			b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)
			
			if j % 10 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_RMS.append(c)
				print("Cost at iteration t=%d, j=%d: %.6f" % (t, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
	
	plt.plot(costs_batch, label="batch")
	plt.plot(costs_RMS, label="rms")
	plt.legend()
	plt.show()
	
if __name__ == '__main__':
	main()

	