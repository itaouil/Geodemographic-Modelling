"""
    ANN implementation (Keras).
"""

import keras                        # DNN library
from preprocessing import *         # Data preprocessing
from keras.models import Sequential # ANN model
from keras.layers import Dense      # ANN layers
from keras.layers import Dropout    # ANN regulatization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Integrate ANN with k-fold
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

# Wrap classifier with GridSearchCV object with
# cross validation implementation
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size": [25, 32], "epochs": [100, 500], "optimizer": ["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Parameters: ", best_parameters)
print("Best accuracy: ", best_accuracy)
