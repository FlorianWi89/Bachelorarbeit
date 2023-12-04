from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, SimpleRNN
import numpy as np
import tensorflow as tf



class RecurrentNeuralNetwork():

    def __init__(self, model_type = 'LSTM', units = 30):
        self.model = Sequential()
        self.units = units
        self.network_type = model_type
        
        self.build_and_compile()
    
    #function to stack the model and compile it
    def build_and_compile(self):

        if self.network_type == 'LSTM':
            self.model.add(LSTM(self.units, return_sequences=True, activation='tanh')),
            self.model.add(Dropout(0.2)),
            self.model.add(LSTM(self.units, return_sequences=False, activation='tanh')),
            self.model.add(Dense(15, activation="linear")),
            self.model.add(Dense(15, activation="tanh")),
            self.model.add(Dense(1))
            
        if self.network_type == 'GRU':
            self.model.add(GRU(self.units, return_sequences=True, activation='tanh'))
            self.model.add(Dropout(0.2)),
            self.model.add(GRU(self.units, return_sequences=False, activation='tanh')),
            self.model.add(Dense(15)),
            self.model.add(Dense(15)),
            self.model.add(Dense(1))
            
        self.model.compile(loss = 'mean_absolute_error',
                   optimizer ='adam',
                   metrics=['mean_squared_error'])
        
        
    
    #fit the models, reshape the data to a 3D Tensor and choose device
    def fit(self, X, y, batch_size=128, epochs=1):
        if X.ndim != 3:
            X = np.array(X).reshape(X.shape[0], X.shape[1], 1)

        if y.ndim != 2:
            y = np.array(y).reshape(y.shape[0],1)

        with tf.device('/cpu:0'):
            self.history = self.model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size)
    
    #let the models predict and return a reshaped output
    def predict(self, X):
        if X.ndim != 3:
            X = np.array(X).reshape(X.shape[0], X.shape[1], 1)

        
        return self.model.predict(X).flatten()
        
    #return some stats for the predictoin errors
    def get_max_mean_min_prediction_error(self, X, y):
        
        errors_in_prediction = np.abs(np.subtract(np.array(self.predict(X)), np.array(y)))

        return np.max(errors_in_prediction), np.mean(errors_in_prediction), np.min(errors_in_prediction)
    
    

    