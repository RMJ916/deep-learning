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
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(500, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01),
                bias_regularizer=keras.regularizers.l1(0.01)),         
    keras.layers.Dense(500, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01),
                bias_regularizer=keras.regularizers.l1(0.01)),           
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(train_images, train_labels, validation_split=0.33, epochs=250, batch_size=240)

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
plt.savefig('A041.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A042.png')
plt.clf()

print('\n *********************Dropout**********************\n')

model_dropout = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
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
plt.savefig('A043.png')
plt.clf()
# summarize history for loss
plt.plot(history_dropout.history['loss'])
plt.plot(history_dropout.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A044.png')
plt.clf()

print('\n *********************Early Stopping**********************\n')

model_es = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(500, activation='relu'),          
    keras.layers.Dense(500, activation='relu'),          
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# simple early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model_es.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# Fit the model
history_es = model_dropout.fit(
    train_images,
    train_labels,
    epochs=250,
    batch_size=240,
    validation_split=0.33,
    callbacks=[es]
)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(history_es.history.keys())

print('\nValidation accuracy:', test_acc)

# summarize history for accuracy
plt.plot(history_es.history['acc'])
plt.plot(history_es.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A045.png')
plt.clf()
# summarize history for loss
plt.plot(history_es.history['loss'])
plt.plot(history_es.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('A046.png')
plt.clf()