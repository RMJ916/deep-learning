# Perceptron Algorithm for synthetic data
from random import seed
from random import randrange
from csv import reader
import numpy as np


w=[1.0]
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	#print("row==========>",row)
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
		#print("Actual ",str(row[-2]))
		#print("Predicted ",str(activation))
	return 1.0 if activation >=0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)

	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	print("predict Y label:",predictions)
	return predictions

def create_dataset():
	w = [1,0,0,0,0,0,0,0,0,0]
	dataset = [np.random.uniform(size=10).tolist() for _ in range(1000)]

	y_data = [np.dot(x,w) for x in dataset]

	for i,x in enumerate(y_data):
		if x < 0.1:
			y_data[i] = -1
		else:
			y_data[i] = 1

	[x.append(y_data[i]) for i,x in enumerate(dataset)]
	#print("y_data========>",y_data)
	return dataset

# Test the Perceptron algorithm on the synthetic data
# load and prepare data

dataset = create_dataset()

# evaluate algorithm
n_folds = 2
l_rate = 0.1
n_epoch = 5
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))