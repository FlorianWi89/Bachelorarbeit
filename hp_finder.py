import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import json
import os
from RecurrentNeuralNetwork import RecurrentNeuralNetwork
from Torch_RNN import Torch_RNN
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MAE
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense

import matplotlib.pyplot as plt

##############################################################

class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):
        super(BaseRNN, self).__init__()
        self.layers = layers

    def forward(self, x):
        pass

    def fit(self, X, y, batch_size=128, epochs=1, lr=0.01, val_split=0.2):

        num_train = int((1-val_split) * len(X))
        X_train = X[: num_train]
        y_train = y[: num_train]

        X_test  = X[num_train : ]
        y_test  = y[num_train :]
        
        # Konvertieren der Daten in PyTorch-Tensoren
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Erstellen von DataLoader-Objekten für das effiziente Laden von Daten während des Trainings
        batch_size = batch_size
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Definition von Verlust und Optimierer
        criterion = nn.L1Loss()  # Mean Absolute Error
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        train_loss_vals= []
        test_loss_vals = []
        
        
        for epoch in range(epochs):
            start_time = time.time()
            self.train(True)
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Ausgabe des Verlusts pro Epoche
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Duration: {round((time.time() - start_time)/60, 3)}')
            train_loss_vals.append(round(loss.item(),4))
            
            val_loss = 0.0
            self.eval()
            
            with torch.no_grad():
                start_time = time.time()
                for batch_X, batch_y in test_loader:
                    pred = self.forward(batch_X)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / batch_size
            #print(val_loss.shape)
            test_loss_vals.append(float(val_loss))
            print(f'Val Loss {val_loss}, Duration: {round((time.time() - start_time)/60, 3)}')
            
        return (train_loss_vals, test_loss_vals)

class SingleLayerRNN(BaseRNN):
    def __init__(self):
        super(SingleLayerRNN, self).__init__(input_size=30, hidden_size=30, output_size=1, layers=1)
        self.lstm1 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out

class DoubleLayerRNN(BaseRNN):
    def __init__(self):
        super(DoubleLayerRNN, self).__init__(input_size=30, hidden_size=30, output_size=1, layers=2)
        self.lstm1 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out

class TripleLayerRNN(BaseRNN):
    def __init__(self):
        super(TripleLayerRNN, self).__init__(input_size=30, hidden_size=30, output_size=1, layers=3)
        self.lstm1 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out

def createRNN(layers=1):
    if layers == 1:
        model = SingleLayerRNN()
    elif layers == 2:
        model = DoubleLayerRNN()
    elif layers == 3:
        model = TripleLayerRNN()
    else:
        raise ValueError("Invalid number of layers. Supported values are 1, 2, or 3.")
    return model


##############################################################

def createLSTM(layers=1, lr = 0.01):
    if layers == 1:
        model = Sequential()
        model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15))
        model.add(tf.keras.layers.Dense(1))
        
    if layers == 2:
        model = Sequential()
        model.add(tf.keras.layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation='tanh'))
        model.add(tf.keras.layers.Dense(1))
        
    if layers == 3:
        model = Sequential()
        model.add(tf.keras.layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation='tanh'))
        model.add(tf.keras.layers.Dense(1))
        
    model.compile(loss = MAE, optimizer =Adam(learning_rate= lr),metrics=[MAE])
    return model

##############################################################


def createGRU(layers=1, lr = 0.01):
    if layers == 1:
        model = Sequential()
        model.add(tf.keras.layers.GRU(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15))
        model.add(tf.keras.layers.Dense(1))
        
    if layers == 2:
        model = Sequential()
        model.add(tf.keras.layers.GRU(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.GRU(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation='tanh'))
        model.add(tf.keras.layers.Dense(1))
        
    if layers == 3:
        model = Sequential()
        model.add(tf.keras.layers.GRU(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.GRU(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.GRU(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation='tanh'))
        model.add(tf.keras.layers.Dense(1))
        
        
    model.compile(loss = MAE, optimizer =Adam(learning_rate= lr),metrics=[MAE])
    return model
        

##############################################################



if __name__ == "__main__":

    scenario = sys.argv[1] #SensorFault or Leakage

    data_size = sys.argv[2] #Percentage of the dataset

    parameter = sys.argv[3] #Layers, LR oder BatchSize

    epochs = int(sys.argv[4]) #number of epochs to train

    sensor_id = int(sys.argv[5])

    #load data

    scenario_type = str(scenario).lower()
    if scenario_type == 'sensorfault':
        scenario_type= 'sensor_fault'

    #load the training data with reduced precision
    data = pd.read_parquet(f'train_data_{scenario_type}.parquet.gzip').to_numpy().astype(np.float32)

    data = data[: int(float(data_size) * len(data))]

    #target_idx = np.random.randint(0,31)
    target_idx = sensor_id

    l = [i for i in  range(0,31)];l.remove(target_idx)

    X, y = data[:,l], data[:,target_idx]

    #output paths
    RNN_path = '/Users/florianwicher/Desktop/Bachelorarbeit/Plots/RNN'
    GRU_path = '/Users/florianwicher/Desktop/Bachelorarbeit/Plots/GRU'
    LSTM_path = '/Users/florianwicher/Desktop/Bachelorarbeit/Plots/LSTM'

    x_plot_data = [i+1 for i in range(epochs)]

    if parameter == 'Layers':
        layers = [1,2,3]

        alpha = 0.01
        batch_size = 256
        


        results = {
            'RNN': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            },
            'GRU': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            },
            'LSTM': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            }
        }

        for i in [1,2,3]:
            LSTM = createLSTM(layers = i, lr=alpha)
            GRU = createGRU(layers = i, lr=alpha)
            RNN = createRNN(layers = i)
            
            
            tloss, vloss = RNN.fit(X, y, batch_size=batch_size, epochs=epochs, lr=alpha, val_split = 0.2)
            LSTM_hist = LSTM.fit(np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(y).reshape(y.shape[0],1), validation_split=0.2, epochs = epochs, batch_size = batch_size)
            GRU_hist = GRU.fit(np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(y).reshape(y.shape[0],1) , validation_split=0.2, epochs = epochs, batch_size = batch_size)
            
            results['RNN']['Train_Loss(MAE)'].append(tloss)
            results['RNN']['Val_Loss(MAE)'].append(vloss)
            
            results['LSTM']['Train_Loss(MAE)'].append(LSTM_hist.history['loss'])
            results['LSTM']['Val_Loss(MAE)'].append(LSTM_hist.history['val_mean_absolute_error'])
            
            results['GRU']['Train_Loss(MAE)'].append(GRU_hist.history['loss'])
            results['GRU']['Val_Loss(MAE)'].append(GRU_hist.history['val_mean_absolute_error'])

            
        with open("Architecture_resuts.json", "w") as outfile: 
                json.dump(results, outfile)

        data = results
        for goal in ['Train', 'Val']:
            # Plot für RNN
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][0], label='1 Layer')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][1], label='2 Layers')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][2], label='3 Layers')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(RNN_path, f'RNN_{goal}_Layers_Loss.png'), bbox_inches="tight")
            #plt.show()

            # Plot für GRU
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][0], label='1 Layer')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][1], label='2 Layers')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][2], label='3 Layers')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(GRU_path, f'GRU_{goal}_Layers_Loss.png'), bbox_inches="tight")
            #plt.show()

            # Plot für LSTM
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][0], label='1 Layer')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][1], label='2 Layers')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][2], label='3 Layers')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(LSTM_path, f'LSTM_{goal}_Layers_Loss.png'), bbox_inches="tight")
            #plt.show()


    if parameter == 'LR':
        batch_size = 256


        results = {
            'RNN': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            },
            'GRU': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            },
            'LSTM': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            }
        }

        for lr in [0.05, 0.01, 0.005, 0.001]:
            LSTM = createLSTM(layers = 2, lr=lr)
            GRU = createGRU(layers = 2, lr=lr)
            RNN = createRNN(layers = 2)
            
            
            tloss, vloss = RNN.fit(X, y, batch_size=batch_size, epochs=epochs, lr=lr, val_split = 0.2)
            LSTM_hist = LSTM.fit(np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(y).reshape(y.shape[0],1), validation_split=0.2, epochs = epochs, batch_size = batch_size)
            GRU_hist = GRU.fit(np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(y).reshape(y.shape[0],1) , validation_split=0.2, epochs = epochs, batch_size = batch_size)
            
            results['RNN']['Train_Loss(MAE)'].append(tloss)
            results['RNN']['Val_Loss(MAE)'].append(vloss)
            
            results['LSTM']['Train_Loss(MAE)'].append(LSTM_hist.history['loss'])
            results['LSTM']['Val_Loss(MAE)'].append(LSTM_hist.history['val_mean_absolute_error'])
            
            results['GRU']['Train_Loss(MAE)'].append(GRU_hist.history['loss'])
            results['GRU']['Val_Loss(MAE)'].append(GRU_hist.history['val_mean_absolute_error'])

        with open("LR_resuts.json", "w") as outfile: 
                json.dump(results, outfile)

        data = results
        for goal in ['Train', 'Val']:
                # Plot für RNN
            plt.figure(figsize=(15, 3))
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][0], label='LR = 0.05')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][1], label='LR = 0.01')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][2], label='LR = 0.005')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][3], label='LR = 0.001')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(RNN_path, f'RNN_{goal}_LR_Loss.png'), bbox_inches="tight")
            #plt.show()

            # Plot für GRU
            plt.figure(figsize=(15, 3))
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][0], label='LR = 0.05')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][1], label='LR = 0.01')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][2], label='LR = 0.005')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][3], label='LR = 0.001')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(GRU_path, f'GRU_{goal}_LR_Loss.png'), bbox_inches="tight")
            #plt.show()

            # Plot für LSTM
            plt.figure(figsize=(15, 3))
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][0], label='LR = 0.05')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][1], label='LR = 0.01')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][2], label='LR = 0.005')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][3], label='LR = 0.001')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(LSTM_path, f'LSTM_{goal}_LR_Loss.png'), bbox_inches="tight")
            #plt.show()

    if parameter == 'BatchSize':


        results = {
            'RNN': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            },
            'GRU': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            },
            'LSTM': {
                'Train_Loss(MAE)' : [],
                'Val_Loss(MAE)' : []
            }
        }

        for batch_size in [32,64, 128, 256, 512]:
            LSTM = createLSTM(layers = 2, lr=0.01)
            GRU = createGRU(layers = 2, lr=0.01)
            RNN = createRNN(layers = 2)
            
            
            tloss, vloss = RNN.fit(X, y, batch_size=batch_size, epochs=epochs, lr=0.01, val_split = 0.2)
            LSTM_hist = LSTM.fit(np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(y).reshape(y.shape[0],1), validation_split=0.2, epochs = epochs, batch_size = batch_size)
            GRU_hist = GRU.fit(np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(y).reshape(y.shape[0],1) , validation_split=0.2, epochs = epochs, batch_size = batch_size)
            
            results['RNN']['Train_Loss(MAE)'].append(tloss)
            results['RNN']['Val_Loss(MAE)'].append(vloss)
            
            results['LSTM']['Train_Loss(MAE)'].append(LSTM_hist.history['loss'])
            results['LSTM']['Val_Loss(MAE)'].append(LSTM_hist.history['val_mean_absolute_error'])
            
            results['GRU']['Train_Loss(MAE)'].append(GRU_hist.history['loss'])
            results['GRU']['Val_Loss(MAE)'].append(GRU_hist.history['val_mean_absolute_error'])

        with open("BatchSize_resuts.json", "w") as outfile: 
                json.dump(results, outfile)
        

        data = results
        for goal in ['Train', 'Val']:
                # Plot für RNN
            plt.figure(figsize=(15, 3))
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][0], label='BS = 32')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][1], label='BS = 64')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][2], label='BS = 128')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][3], label='BS = 256')
            plt.plot(data['RNN'][f'{goal}_Loss(MAE)'][4], label='BS = 512')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(RNN_path, f'RNN_{goal}_BatchSize_Loss.png'), bbox_inches="tight")
            #plt.show()

            # Plot für GRU
            plt.figure(figsize=(15, 3))
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][0], label='BS = 32')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][1], label='BS = 64')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][2], label='BS = 128')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][3], label='BS = 256')
            plt.plot(data['GRU'][f'{goal}_Loss(MAE)'][4], label='BS = 512')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(GRU_path, f'GRU_{goal}_BatchSize_Loss.png'), bbox_inches="tight")
            #plt.show()

            # Plot für LSTM
            plt.figure(figsize=(15, 3))
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][0], label='BS = 32')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][1], label='BS = 64')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][2], label='BS = 128')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][3], label='BS = 256')
            plt.plot(data['LSTM'][f'{goal}_Loss(MAE)'][4], label='BS = 512')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(LSTM_path, f'LSTM_{goal}_BatchSize_Loss.png'), bbox_inches="tight")
            #plt.show()