# USAGE
# python sgd.py
# From my view this method is vital for linear regression because it will divide whole data in small chunks and do the classification so it will speed up the process of computation

# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

# def create_dataset():
# 	#w = [1,0,0,0,0,0,0,0,0,0]
# 	#dataset = [np.random.uniform(size=10).tolist() for _ in range(1000)]
# 	X = [np.random.uniform(size=10).tolist() for _ in range(100)]
#
# 	#print("x==========>",X)
# 	#y = [np.dot(x,w) for x in X]
# 	y = [(np.dot(i+1,X[i]) + 0.1)  for i in range(len(X))]
# 	#print(y)
# 	Y = []
# 	for i in range(len(y)):
# 		sum = 0
# 		for j in y[i]:
# 			sum = sum + j
# 		Y.append(sum)
# 		sum = 0
# 	print(Y)
# 	print(len(Y))
#
# 	print(len(y))
# 	#print("y============>",y)
# 	for i,x in enumerate(y):
# 		if x < 0.1:
# 			y[i] = -1
# 		else:
# 			y[i] = 1
#
# 	#[x.append(y[i]) for i,x in enumerate(dataset)]
# 	#print("y_data========>",y_data)
# 	return (X,y)

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-x))

def next_batch(X, y, batchSize):
	# loop over our dataset `X` in mini-batches of size `batchSize`
	#print("random===>",np.arange(0, X.shape[0], batchSize))
	for i in np.arange(0, X.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (X[i:i + batchSize], y[i:i + batchSize])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 400 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=100, n_features=10, centers=2,
	cluster_std=2.5, random_state=95)

#print("X============>",X)
#print("y============>",y)
# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]

#print("updated X========>",X)
# initialize our weight matrix such it has the same number of
# columns as our input features
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))
#print("weights=====>",W)

# initialize a list to store the loss value for each epoch
lossHistory = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	# initialize the total loss for the epoch
	epochLoss = []

	# loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
		# take the dot product between our current batch of
		# features and weight matrix `W`, then pass this value
		# through the sigmoid activation function
		preds = sigmoid_activation(batchX.dot(W))

		# now that we have our predictions, we need to determine
		# our `error`, which is the difference between our predictions
		# and the true values
		error = preds - batchY

		# given our `error`, we can compute the total loss value on
		# the batch as the sum of squared loss
		loss = np.sum(error ** 2)
		epochLoss.append(loss)


		# the gradient update is therefore the dot product between
		# the transpose of our current batch and the error on the
		# # batch
		gradient = batchX.T.dot(error) / batchX.shape[0]

		# use the gradient computed on the current batch to take
		# a "step" in the correct direction
		W += -args["alpha"] * gradient

	# update our loss history list by taking the average loss
	# across all batches
	lossHistory.append(np.average(epochLoss))
	print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))

#print("ram==========>",np.random.choice(1000, 10))
for i in np.random.choice(100, 10):
	# compute the prediction by taking the dot product of the
	# current feature vector with the weight matrix W, then
	# passing it through the sigmoid activation function
	activation = sigmoid_activation(X[i].dot(W))

	# the sigmoid function is defined over the range y=[0, 1],
	# so we can use 0.5 as our threshold -- if `activation` is
	# below 0.5, it's class `0`; otherwise it's class `1`
	label = 0 if activation < 0.5 else 1

	# show our output classification
	print("activation={:.4f}; predicted_label={}, true_label={}".format(
		activation, label, y[i]))
# compute the line of best fit by setting the sigmoid function
# to 0 and solving for X2 in terms of X1
Y = (-W[0] - (W[1] * X)) / W[2]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
# fig = plt.figure()
# plt.plot(np.arange(0, args["epochs"]), lossHistory)
# fig.suptitle("Training Loss")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
plt.show()
