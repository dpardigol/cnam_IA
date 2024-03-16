# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:13:09 2024

@author: david
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import keras
from keras import Sequential
from keras.layers import Dense, Activation

class CustomEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 input_dim=10,
                 output_dim=6,
                 neurons_per_layer=30,
                 num_layers=1, 
                 optimizer=keras.optimizers.SGD(learning_rate=0.5), 
                 activation='relu', 
                 epochs=10, 
                 batch_size=32, 
                 verbose=0):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.build_model()

    def build_model(self):
        self._model = Sequential()
        self._model.add(Dense(self.neurons_per_layer, 
                              input_dim=self.input_dim, 
                              name='fc0', 
                              kernel_initializer='glorot_uniform', 
                              bias_initializer='zeros'))
        self._model.add(Activation(self.activation))
        for k in range(self.num_layers-1):
            self._model.add(Dense(self.neurons_per_layer,
                                  input_dim = self.neurons_per_layer, 
                                  name=f"fc{k+1}", 
                                  kernel_initializer='glorot_uniform', 
                                  bias_initializer='zeros'))
            self._model.add(Activation(self.activation))
        self._model.add(Dense(self.output_dim,
                              input_dim = self.neurons_per_layer, 
                              name='fclast', 
                              kernel_initializer='glorot_uniform', 
                              bias_initializer='zeros'))
        self._model.add(Activation('softmax'))

        self._model.compile(loss='categorical_crossentropy', 
                            optimizer=self.optimizer, 
                            metrics=['accuracy'])

    def fit(self, X, y):
        self._model.fit(X, 
                        y, 
                        batch_size=self.batch_size, 
                        epochs=self.epochs, 
                        verbose=self.verbose)
        self._is_fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self._model.predict(X)

    def score(self, X, y):
        check_is_fitted(self)
        return self._model.evaluate(X, 
                                    y,
                                    verbose = self.verbose)[1]

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted
