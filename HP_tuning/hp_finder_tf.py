import pandas as pd
import numpy as np

import time
import sys
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf

##############################################################

def createRNN(layers=1, lr = 0.001):
    model = tf.keras.Sequential()
    if layers == 1:
        
        model.add(tf.keras.layers.SimpleRNN(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

    elif layers == 2:
        
        model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.SimpleRNN(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

    elif layers == 3:
        
        model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.SimpleRNN(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))
    
    model.compile(loss = 'mean_absolute_error',
                   optimizer =tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                   metrics=['mean_absolute_error'])
    return model

##############################################################


def createGRU(layers=1, lr = 0.001):
    model = tf.keras.Sequential()

    if layers == 1:

        model.add(tf.keras.layers.GRU(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

    elif layers == 2:
        
        model.add(tf.keras.layers.GRU(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.GRU(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

    elif layers == 3:
        
        model.add(tf.keras.layers.GRU(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.GRU(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.GRU(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))
    
    model.compile(loss = 'mean_absolute_error',
                   optimizer =tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                   metrics=['mean_absolute_error'])
    return model

##############################################################

def createLSTM(layers=1, lr = 0.001):
    model = tf.keras.Sequential()
    
    if layers == 1:

        model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

    elif layers == 2:
        
        model.add(tf.keras.layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

    elif layers == 3:
        
        model.add(tf.keras.layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='tanh'))
        model.add(tf.keras.layers.Dense(15, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))
    
    model.compile(loss = 'mean_absolute_error',
                   optimizer =tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                   metrics=['mean_absolute_error'])
    return model


##############################################################



if __name__ == "__main__":

    scenario = sys.argv[1] #SensorFault or Leakage

    data_size = sys.argv[2] #Percentage of the dataset

    parameter = sys.argv[3] #Layers, LR oder BatchSize

    epochs = int(sys.argv[4]) #number of epochs to train

    sensor_id = int(sys.argv[5])

    #load data
    scenario = str(scenario)
    scenario_type = str(scenario).lower()
    if scenario_type == 'sensorfault':
        scenario_type= 'sensor_fault'

    #load the training data with reduced precision
    data = pd.read_parquet(f'../Data/train_data_{scenario_type}.parquet.gzip').to_numpy().astype(np.float64)

    data = data[: int(float(data_size) * len(data))]

    #target_idx = np.random.randint(0,31)
    target_idx = sensor_id

    l = [i for i in  range(0,31)];l.remove(target_idx)

    X, y = data[:,l], data[:,target_idx]

    X = np.array(X).reshape(X.shape[0], X.shape[1], 1)
    y = np.array(y).reshape(y.shape[0],1)

    #output paths
    

    x_plot_data = [i+1 for i in range(epochs)]

    result_template = {
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

    if parameter == 'Layers':
        layers = [1,2,3]
        batch_size = 128
        


        results = result_template.copy()

        for i in [1,2,3]:
            LSTM = createLSTM(layers = i)
            GRU = createGRU(layers = i)
            RNN = createRNN(layers = i)
            
            with tf.device('/cpu:0'):
                history = RNN.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['RNN']['Train_Loss(MAE)'].append(history.history['loss'])
            results['RNN']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])
            
            with tf.device('/cpu:0'):
                history = GRU.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['GRU']['Train_Loss(MAE)'].append(history.history['loss'])
            results['GRU']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])
            
            with tf.device('/cpu:0'):
                history = LSTM.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['LSTM']['Train_Loss(MAE)'].append(history.history['loss'])
            results['LSTM']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])

            
        with open(f"Architecture_results_{scenario}.json", "w") as outfile: 
                json.dump(results, outfile)

        data = results
        for goal in ['Train', 'Val']:
            # Plot für RNN
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][0], label='1 Layer', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][1], label='2 Layers', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][2], label='3 Layers', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_RNN_{goal}_Layers_Loss.png', bbox_inches="tight")
            #plt.show()

            # Plot für GRU
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][0], label='1 Layer', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][1], label='2 Layers', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][2], label='3 Layers', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_GRU_{goal}_Layers_Loss.png', bbox_inches="tight")
            #plt.show()

            # Plot für LSTM
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][0], label='1 Layer', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][1], label='2 Layers', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][2], label='3 Layers', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_LSTM_{goal}_Layers_Loss.png', bbox_inches="tight")
            #plt.show()


    if parameter == 'LR':
        batch_size = 128


        results = result_template.copy()

        for lr in [0.05, 0.01, 0.005, 0.001]:
            
            LSTM = createLSTM(layers = 2, lr=lr)
            GRU = createGRU(layers = 2, lr=lr)
            RNN = createRNN(layers = 2, lr=lr)
            
            
            with tf.device('/cpu:0'):
                history = RNN.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['RNN']['Train_Loss(MAE)'].append(history.history['loss'])
            results['RNN']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])
            
            with tf.device('/cpu:0'):
                history = GRU.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['GRU']['Train_Loss(MAE)'].append(history.history['loss'])
            results['GRU']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])
            
            with tf.device('/cpu:0'):
                history = LSTM.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['LSTM']['Train_Loss(MAE)'].append(history.history['loss'])
            results['LSTM']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])

        with open(f"LR_results_{scenario}.json", "w") as outfile: 
                json.dump(results, outfile)

        data = results
        for goal in ['Train', 'Val']:
                # Plot für RNN
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][0], label='LR = 0.05', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][1], label='LR = 0.01', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][2], label='LR = 0.005', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][3], label='LR = 0.001', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_RNN_{goal}_LR_Loss.png', bbox_inches="tight")
            #plt.show()

            # Plot für GRU
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][0], label='LR = 0.05', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][1], label='LR = 0.01', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][2], label='LR = 0.005', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][3], label='LR = 0.001', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_GRU_{goal}_LR_Loss.png', bbox_inches="tight")
            #plt.show()

            # Plot für LSTM
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][0], label='LR = 0.05', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][1], label='LR = 0.01', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][2], label='LR = 0.005', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][3], label='LR = 0.001', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_LSTM_{goal}_LR_Loss.png', bbox_inches="tight")
            #plt.show()

    if parameter == 'BatchSize':


        results = result_template.copy()

        for batch_size in [32,64, 128, 256, 512]:
            LSTM = createLSTM(layers = 2)
            GRU = createGRU(layers = 2)
            RNN = createRNN(layers = 2)
            
            
            with tf.device('/cpu:0'):
                history = RNN.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['RNN']['Train_Loss(MAE)'].append(history.history['loss'])
            results['RNN']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])
            
            with tf.device('/cpu:0'):
                history = GRU.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['GRU']['Train_Loss(MAE)'].append(history.history['loss'])
            results['GRU']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])
            
            with tf.device('/cpu:0'):
                history = LSTM.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split = 0.10)
            results['LSTM']['Train_Loss(MAE)'].append(history.history['loss'])
            results['LSTM']['Val_Loss(MAE)'].append(history.history['val_mean_absolute_error'])

        with open(f"BatchSize_results_{scenario}.json", "w") as outfile: 
                json.dump(results, outfile)
        

        data = results
        for goal in ['Train', 'Val']:
                # Plot für RNN
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][0], label='BS = 32', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][1], label='BS = 64', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][2], label='BS = 128', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][3], label='BS = 256', marker='.')
            plt.plot(x_plot_data, data['RNN'][f'{goal}_Loss(MAE)'][4], label='BS = 512', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_RNN_{goal}_BatchSize_Loss.png', bbox_inches="tight")
            #plt.show()

            # Plot für GRU
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][0], label='BS = 32', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][1], label='BS = 64', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][2], label='BS = 128', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][3], label='BS = 256', marker='.')
            plt.plot(x_plot_data, data['GRU'][f'{goal}_Loss(MAE)'][4], label='BS = 512', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_GRU_{goal}_BatchSize_Loss.png', bbox_inches="tight")
            #plt.show()

            # Plot für LSTM
            plt.figure(figsize=(15, 3))
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][0], label='BS = 32', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][1], label='BS = 64', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][2], label='BS = 128', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][3], label='BS = 256', marker='.')
            plt.plot(x_plot_data, data['LSTM'][f'{goal}_Loss(MAE)'][4], label='BS = 512', marker='.')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('MAE Loss')
            plt.legend(loc='upper right')
            plt.savefig(f'Plots/{scenario}_LSTM_{goal}_BatchSize_Loss.png', bbox_inches="tight")
            #plt.show()