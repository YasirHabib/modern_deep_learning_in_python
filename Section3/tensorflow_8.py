# Section 8, Lecture 28

import numpy as np
import tensorflow as tf
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
    # add an extra layer just for fun
	M1 = 300
	M2 = 100
	K = len(set(Ytrain))
	
	W1_init = np.random.randn(D, M1) / np.sqrt(D)
	b1_init = np.zeros(M1)
	W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
	b2_init = np.zeros(M2)
	W3_init = np.random.randn(M2, K) / np.sqrt(M2)
	b3_init = np.zeros(K)
	
	lr = 0.00004
	reg = 0.01
	batch_sz = 500
	n_batches = N // batch_sz
	epochs = 15
	
	# define variables and expressions
	X = tf.placeholder(tf.float32, shape=(None, D), name='X')
	T = tf.placeholder(tf.float32, shape=(None, K), name='T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))
	
	# define the model
	Z1 = tf.nn.relu(tf.matmul(X, W1)+b1)
	Z2 = tf.nn.relu(tf.matmul(Z1, W2)+b2)
	pY = tf.matmul(Z2, W3)+b3 # remember, the cost function does the softmaxing!
	
	# define the cost function
	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=pY))
	
    # we choose the optimizer but don't implement the algorithm ourselves
    # let's go with RMSprop, since we just learned about it.
    # it includes momentum!
	train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
	
	prediction = tf.argmax(pY, axis=1)
	
	costs_batch = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(epochs):
			tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
			for j in range(n_batches):
				x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
				y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
			
				session.run(train_op, feed_dict={X: x, T: y})
				if j % 50 == 0:
					cost_val = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
					prediction_val = session.run(prediction, feed_dict={X: Xtest})
					e = error_rate(prediction_val, Ytest)
					print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, e))
					costs_batch.append(cost_val)

	plt.plot(costs_batch)
	plt.show()

if __name__ == '__main__':
	main()