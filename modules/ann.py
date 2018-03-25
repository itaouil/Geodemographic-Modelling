"""
    ANN implementation (Keras).
"""

import keras                        # DNN library
from preprocessing import *         # Data preprocessing
from keras.models import Sequential # ANN model
from keras.layers import Dense      # ANN layers
from keras.layers import Dropout    # ANN regulatization

# Initialise model object
classifier = Sequential()

# Input layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# Hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# Output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Apply stochastic gradient descent
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit ANN to training set
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# Make predictions & Confusion matrix
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)
