# -*- coding: utf-8 -*-
"""
multiclass logistic regression
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import tensorflow as tf

(train_img,train_lbl),(test_img,test_lbl)=tf.keras.datasets.mnist.load_data()
train_img=train_img.reshape(60000,784)
test_img=test_img.reshape(10000,784)

# all parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it
logisticRegr = LogisticRegression(tol=0.01,penalty='l2',solver = 'lbfgs',C=0.01,max_iter=100,multi_class='multinomial')
logisticRegr.fit(train_img, train_lbl)

# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))

# Predict for multiple Observation (image)
# make prediction on entire test data
predictions = logisticRegr.predict(test_img)
# Use score method to get accuracy of model
score = logisticRegr.score(test_img, test_lbl)
print(score)
#confusion metrix
cm = metrics.confusion_matrix(test_lbl, predictions)
print(cm)
#Using Matplotlib
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x),
  horizontalalignment='center',
  verticalalignment='center')
plt.show()