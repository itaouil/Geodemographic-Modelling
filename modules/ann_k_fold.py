"""
    ANN implementation (Keras).
"""

import keras                        # DNN library
from preprocessing import *         # Data preprocessing
from keras.models import Sequential # ANN model
from keras.layers import Dense      # ANN layers
from keras.layers import Dropout    # ANN regulatization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Integrate ANN with k-fold
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dropout(p=0.1))
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
