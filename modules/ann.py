"""
    ANN implementation (Keras).
"""

import keras                        # DNN library
from keras.models import Sequential # ANN model
from keras.layers import Dense      # ANN layers
from preprocessing import *

# # Initialise model object
# classifier = Sequential()
#
# # Input layer
# classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
#
# # Hidden layer
# classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
#
# # Output layer
# classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
#
# # Apply stochastic gradient descent
# classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#
# # Fit ANN to training set
# classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# Make predictions & Confusion matrix
# y_pred = classifier.predict(x_test)
# y_pred = (y_pred > 0.5)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# K-Fold validation class and Keras wrapper
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Integrate ANN with k-fold
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)

# Compute mean of accuracies
mean = accuracies.mean()
variance = accuracies.std()
print("Mean: ", mean)
print("Variance: ", variance)
