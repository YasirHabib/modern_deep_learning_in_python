# Section 4, Lecture 13
	
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
	
	lr = 0.001
	reg = 0.01
	batch_sz = 500
	n_batches = N // batch_sz
	epochs = 10
	beta1 = 0.9
	beta2 = 0.999
	eps = 1e-8
	
	# 1st moment
	mW2 = 0
	mb2 = 0
	mW1 = 0
	mb1 = 0

	# 2nd moment
	vW1 = 0
	vb1 = 0
	vW2 = 0
	vb2 = 0
	
	# 1. Adam
	costs_adam = []
	t = 1
	for i in range(epochs):
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
			
			# new m
			mW2 = beta1 * mW2 + (1 - beta1) * gW2
			mb2 = beta1 * mb2 + (1 - beta1) * gb2
			mW1 = beta1 * mW1 + (1 - beta1) * gW1
			mb1 = beta1 * mb1 + (1 - beta1) * gb1
			
			# new v
			vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
			vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
			vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
			vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
			
			# bias correction
			correction1 = 1 - beta1 ** t
			hat_mW2 = mW2 / correction1
			hat_mb2 = mb2 / correction1
			hat_mW1 = mW1 / correction1
			hat_mb1 = mb1 / correction1
			
			correction2 = 1 - beta2 ** t
			hat_vW2 = vW2 / correction2
			hat_vb2 = vb2 / correction2
			hat_vW1 = vW1 / correction2
			hat_vb1 = vb1 / correction2
			
			# update t
			t += 1
			
			# apply updates to the params
			W2 -= lr * hat_mW2 / np.sqrt(hat_vW2 + eps)
			b2 -= lr * hat_mb2 / np.sqrt(hat_vb2 + eps)
			W1 -= lr * hat_mW1 / np.sqrt(hat_vW1 + eps)
			b1 -= lr * hat_mb1 / np.sqrt(hat_vb1 + eps)			
			
			if j % 10 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_adam.append(c)
				print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
	print("\n")
	
	# 2. RMSprop with momentum
	W1 = W1_0.copy()
	b1 = b1_0.copy()
	W2 = W2_0.copy()
	b2 = b2_0.copy()
	
	# rmsprop cache
	cache_W2 = 1
	cache_b2 = 1
	cache_W1 = 1
	cache_b1 = 1
	decay_rate = 0.999

	# momentum
	mu = 0.9
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0
	
	costs_RMS = []
	for i in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			pY, Z = forward(x, W1, b1, W2, b2)
			
			# updates
			gW2 = (derivative_W2(Z, pY, y) + reg*W2)
			cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
			dW2 = mu * dW2 + (1 - mu) * lr * gW2 / (np.sqrt(cache_W2) + eps)
			W2 -= dW2

			gb2 = (derivative_b2(pY, y) + reg*b2)
			cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
			db2 = mu * db2 + (1 - mu) * lr * gb2 / (np.sqrt(cache_b2) + eps)
			b2 -= db2

			gW1 = (derivative_W1(x, W2, Z, pY, y) + reg*W1)
			cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
			dW1 = mu * dW1 + (1 - mu) * lr * gW1 / (np.sqrt(cache_W1) + eps)
			W1 -= dW1

			gb1 = (derivative_b1(W2, Z, pY, y) + reg*b1)
			cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
			db1 = mu * db1 + (1 - mu) * lr * gb1 / (np.sqrt(cache_b1) + eps)
			b1 -= db1
			
			if j % 10 == 0:
				pY_test, _ = forward(Xtest, W1, b1, W2, b2)
				c = cost(pY_test, Ytest_ind)
				costs_RMS.append(c)
				print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, c))

				e = error_rate(pY_test, Ytest)
				print("Error rate:", e)
				
	plt.plot(costs_adam, label='adam')
	plt.plot(costs_RMS, label='rmsprop')
	plt.legend()
	plt.show()

if __name__ == '__main__':
    main()