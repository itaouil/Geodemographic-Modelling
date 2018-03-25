# **Geodemographic-Modelling**
Given a dataset from a bank, containing clients data (age, salary, bank products used, etc) create a model able to categorise which customer is more likely to leave the bank.

## **Architecture**
Given the small dataset and the rather limited number of independent variables to train on, a deep/wide architecture is not really necessary. Hence, why a three layer ANN with an input layer, a hidden layer and an output one has been chosen, with respectively 6 neurons for the first two layers and 1 neuron for the output layer (making a binary classification taks, either 0 or 1).

To implement the architecture the **Keras** library with **Tensorflow** backend was used, in particular its Sequential and Dense modules for the creation of the ANN model and its layers.

## **Optimisation**
In the modules folder you will find three Python scripts. All of the scripts use the preprocessing file to load the dataset, retrieve the required independent variables, encode the categorical data (as ANNs only work on numerical values), one hot encode the dummy variables and split the dataset in training and testing.

To optimise the ANN (although the dataset is quite simple) there are few approaches, a good one is to use K-Fold cross validation to validate the model and avoid overfitting, another approach consists in using dropout regularisation which entails randomly disabling some neurons in any layer specified such that no dependency is built between input variables and specific neurons. The last approach tried consists in parameters tuning. In fact, being the ANN model a parametric model, we aim at optimising these parameters as much as possible to obtain the best accuracy possible.

In terms of implementation, such optimisation have made use of the Keras dropout and scikit_learn wrapper as well as pure scikit_learn functions (GridSearchCV, cross_val_score, etc).
