# Section 4, Lecture 9

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utility import get_transformed_data, y2indicator, cost, error_rate
from mlp import forward, derivative_W2, derivative_b2, derivative_W1, derivative_b1

def main():
	# compare 3 scenarios:
    # 1. batch SGD
    # 2. batch SGD with momentum
    # 3. batch SGD with Nesterov momentum
	
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
			
			if j % 50 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_batch.append(c)
				print("Cost at iteration t=%d, j=%d: %.6f" % (t, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
	print("\n")
			
	# 2. batch with momentum
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	
	mu = 0.9
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0
	
	costs_batch_momentum = []
	for t in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			pY, Z = forward(x, W1, b1, W2, b2)
			
			# gradients
			gW2 = (derivative_W2(Z, pY, y) + reg*W2)
			gb2 = (derivative_b2(pY, y) + reg*b2)
			gW1 = (derivative_W1(x, W2, Z, pY, y) + reg*W1)
			gb1 = (derivative_b1(W2, Z, pY, y) + reg*b1)
			
			# update velocities
			dW2 = mu*dW2 - lr*gW2
			db2 = mu*db2 - lr*gb2
			dW1 = mu*dW1 - lr*gW1
			db1 = mu*db1 - lr*gb1
			
			# updates
			W2 += dW2
			b2 += db2
			W1 += dW1
			b1 += db1
			
			if j % 50 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_batch_momentum.append(c)
				print("Cost at iteration t=%d, j=%d: %.6f" % (t, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
	print("\n")
				
	# 3. batch with Nesterov momentum
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	
	mu = 0.9
	vW2 = 0
	vb2 = 0
	vW1 = 0
	vb1 = 0
	
	costs_batch_momentum_nesterov = []
	for t in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			pY, Z = forward(x, W1, b1, W2, b2)
			
			# gradients
			gW2 = (derivative_W2(Z, pY, y) + reg*W2)
			gb2 = (derivative_b2(pY, y) + reg*b2)
			gW1 = (derivative_W1(x, W2, Z, pY, y) + reg*W1)
			gb1 = (derivative_b1(W2, Z, pY, y) + reg*b1)
			
			# v update
			vW2 = mu*vW2 - lr*gW2
			vb2 = mu*vb2 - lr*gb2
			vW1 = mu*vW1 - lr*gW1
			vb1 = mu*vb1 - lr*gb1
			
			# param update
			W2 += mu*vW2 - lr*gW2
			b2 += mu*vb2 - lr*gb2
			W1 += mu*vW1 - lr*gW1
			b1 += mu*vb1 - lr*gb1
				
			if j % 50 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_batch_momentum_nesterov.append(c)
				print("Cost at iteration t=%d, j=%d: %.6f" % (t, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
	
	plt.plot(costs_batch, label="batch")
	plt.plot(costs_batch_momentum, label="momentum")
	plt.plot(costs_batch_momentum_nesterov, label="nesterov")
	plt.legend()
	plt.show()
	
if __name__ == '__main__':
	main()
