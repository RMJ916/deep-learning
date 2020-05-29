# tf.keras

# Helper libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0

print('\n *********************L2 Regulizer**********************\n')

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01),
                bias_regularizer=keras.regularizers.l1(0.01)),         
    keras.layers.Dense(500, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01),
                bias_regularizer=keras.regularizers.l1(0.01)),           
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images.reshape(-1,28, 28, 1)
test_images = test_images.reshape(-1,28, 28, 1)

history=model.fit(train_images, train_labels, epochs=250,validation_split=0.13)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(history.history.keys())

print('\nValidation accuracy:', test_acc)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A04L21.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A04L22.png')
plt.clf()

print('\n *********************Dropout**********************\n')

model_dropout = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.2), 
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.5),           
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.5),            
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model_dropout.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

train_images = train_images.reshape(-1,28, 28, 1)
test_images = test_images.reshape(-1,28, 28, 1)

# Fit the model
history_dropout = model_dropout.fit(
    train_images,
    train_labels,
    epochs=250,
    batch_size=240,
    validation_split=0.33
)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(history_dropout.history.keys())

print('\nValidation accuracy:', test_acc)

# summarize history for accuracy
plt.plot(history_dropout.history['acc'])
plt.plot(history_dropout.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A04DR1.png')
plt.clf()
# summarize history for loss
plt.plot(history_dropout.history['loss'])
plt.plot(history_dropout.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A04DR2.png')
plt.clf()