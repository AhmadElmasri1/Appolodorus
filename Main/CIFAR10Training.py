import pandas
import numpy
import tensorflow
import matplotlib
import keras
import cv2
from keras.callbacks import callbacks
import sklearn
import sys

sys.path.insert(0, 'Utility/')
import ImageAndLabelPreparation

# (xTraining, yTraining), (xTesting, yTesting) = keras.datasets.cifar10.load_data()

(xTraining, yTraining) = ImageAndLabelPreparation.LoadDataSet

print('xTraining Set:')
print(xTraining.shape)
print(xTraining[0].shape)
print()

print('yTraining Set:')
print(yTraining.shape)
print(yTraining[0].shape)
print(yTraining[:20])
print()

xTraining = xTraining / 255
# xTesting = xTesting / 255

yCategoricalTraining = tensorflow.keras.utils.to_categorical(yTraining)
# yCategoricalTesting = tensorflow.keras.utils.to_categorical(yTesting)

# model = keras.models.load_model('../Models/model.h5')
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 32, kernel_size=(4,4), input_shape=(50,100,3), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512, activation = 'relu'))

model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])

model.summary()

earlyStopping = keras.callbacks.EarlyStopping(monitor='loss',patience=2)


model.fit(xTraining, yCategoricalTraining, epochs=2,
          validation_data=(xTesting,yCategoricalTesting ))

# serialize model to JSON
model.save("../Models/model.h5")
print("Saved model to disk")

#Evaluation

metrics = pandas.DataFrame(model.history.history)

print('metrics: ' )
print(metrics.columns)

predictions = model.predict_classes(xTesting)


# print(sklearn.metrics.classification_report)
#
# cv2.imshow('Preview', xTraining[21])
# cv2.waitKey(0)

