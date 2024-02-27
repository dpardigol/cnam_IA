# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:02:41 2024

@author: david
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)

path_variable = Path(r'C:\Users\david\OneDrive\Bureau\CNAM_IA\RCP209\Projet\data')

train_data = path_variable / "train.csv"
test_data = path_variable / "test.csv"

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)

full_data = pd.concat([df_train,df_test])

X_train, X_test, y_train, y_test = train_test_split(full_data.iloc[:,:-1], full_data['Activity'], test_size=0.3)


class KerasClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 imput_dim = 10,
                 neurons_per_layer=30,
                 num_layers=1, 
                 optimizer='sgd', 
                 activation='relu', 
                 epochs=10, 
                 batch_size=32, 
                 verbose=0):
        self.imput_dim = imput_dim
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(neurons_per_layer, input_dim=100, name='fc0', kernel_initializer='glorot_uniform',bias_initializer='zero'))
        model.add(Activation(activation))
        for k in range(num_layers-1):
            model.add(Dense(neurons_per_layer, name=f"fc{k+1}", kernel_initializer='glorot_uniform',bias_initializer='zero'))
            model.add(Activation(activation))
        model.add(Dense(6, name='fclast', kernel_initializer='glorot_uniform',bias_initializer='zero'))
        model.add(Activation('softmax'))

    def fit(self, X, y):
        print("model is being fitted")
        # Use categorical crossentropy for multiclass classification
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        Y = keras.utils.to_categorical(y, 6)
        self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        # Return class labels directly
        return np.argmax(self.model.predict(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)






def model(optimizer         = tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
          num_layers        = 1,
          batch_size        = 200,
          neurons_per_layer = 30,
          activation        = 'relu',
          epochs            = 10):
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=100, name='fc0', kernel_initializer='glorot_uniform',bias_initializer='zero'))
    model.add(Activation(activation))
    for k in range(num_layers-1):
        model.add(Dense(neurons_per_layer, name=f"fc{k+1}", kernel_initializer='glorot_uniform',bias_initializer='zero'))
        model.add(Activation(activation))
    model.add(Dense(6, name='fclast', kernel_initializer='glorot_uniform',bias_initializer='zero'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train_pca, keras.utils.to_categorical(y_train_digits,6), batch_size=batch_size, epochs=epochs, verbose=1)
    scores = model.evaluate(X_test_pca, keras.utils.to_categorical(y_test_digits,6), verbose=0)
    print(f"{model.metrics_names[0]}: {scores[0]*100:.2f}")
    print(f"{model.metrics_names[1]}: {scores[1]*100:.2f}")
    return scores[1]*100















