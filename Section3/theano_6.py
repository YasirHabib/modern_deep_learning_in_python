# Section 7, Lecture 25

# A 1-hidden-layer neural network in Theano.
# This code is not optimized for speed.
# It's just to get something working, using the principles we know.

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utility import get_transformed_data, y2indicator

def error_rate(prediction, Y):
	return np.mean(prediction != Y)

def main():
	# step 1: get the data and define all the usual variables
	Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
	
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)
	
	N, D = Xtrain.shape
	M = 300
	K = len(set(Ytrain))
	
	W1_init = np.random.randn(D, M) / np.sqrt(D)
	b1_init = np.zeros(M)
	W2_init = np.random.randn(M, K) / np.sqrt(M)
	b2_init = np.zeros(K)
	
	lr = 0.00004
	reg = 0.01
	batch_sz = 500
	n_batches = N // batch_sz
	epochs = 20
	
	# step 2: define theano variables and expressions
	thX = T.matrix('X')
	thT = T.matrix('T')
	W1 = theano.shared(W1_init, 'W1')
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W2_init, 'W2')
	b2 = theano.shared(b2_init, 'b2')
	
    # we can use the built-in theano functions to do relu and softmax
	thZ = T.nnet.relu(thX.dot(W1) + b1)
	thpY = T.nnet.softmax(thZ.dot(W2) + b2)
	
	# define the cost function and prediction
	cost = -(thT * T.log(thpY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
	prediction = T.argmax(thpY, axis=1)
	
	# step 3: training expressions and functions
	update_W1 = W1 - lr * T.grad(cost, W1)
	update_b1 = b1 - lr * T.grad(cost, b1)
	update_W2 = W2 - lr * T.grad(cost, W2)
	update_b2 = b2 - lr * T.grad(cost, b2)
	
	train = theano.function(inputs = [thX, thT], updates = [(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)])
	
    # create another function for this because we want it over the whole dataset
	get_prediction = theano.function(inputs = [thX, thT], outputs = [cost, prediction])
	
	costs_batch = []
	for i in range(epochs):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
			train(x, y)
			if j % 10 == 0:
				cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
				e = error_rate(prediction_val, Ytest)
				print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, e))
				costs_batch.append(cost_val)

	plt.plot(costs_batch)
	plt.show()
	
if __name__ == '__main__':
    main()