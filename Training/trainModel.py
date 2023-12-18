import pandas as pd
import numpy as np
from FaultDetector import EnsembleSystem
from Models.RecurrentNeuralNetwork import RecurrentNeuralNetwork
from Models.LinearModel import LinearModel
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#function to get the trained models
def train_model(train_data, flow_nodes, pressure_nodes, model_type = 'LSTM', batch_size=64, epochs=1):

    if model_type == 'LSTM':
        
        model_ensemble = EnsembleSystem(RecurrentNeuralNetwork, flow_nodes, pressure_nodes, model_type='LSTM')
        model_ensemble.fit_seq(train_data, batch_size, epochs)
        return model_ensemble

    if model_type == 'GRU':

        model_ensemble = EnsembleSystem(RecurrentNeuralNetwork, flow_nodes, pressure_nodes, model_type='GRU')
        model_ensemble.fit_seq(train_data, batch_size, epochs)
        return model_ensemble

    if model_type == 'RNN':

        model_ensemble = EnsembleSystem(RecurrentNeuralNetwork, flow_nodes, pressure_nodes, model_type='RNN')
        model_ensemble.fit_seq(train_data, batch_size, epochs)
        return model_ensemble

    if model_type == 'LinearRegression':

        model_ensemble = EnsembleSystem(LinearModel, flow_nodes, pressure_nodes, model_type='LinearRegression')
        model_ensemble.fit_seq(train_data, batch_size, epochs)
        return model_ensemble

    if model_type == 'Ridge':

        model_ensemble = EnsembleSystem(LinearModel, flow_nodes, pressure_nodes, model_type='Ridge')
        model_ensemble.fit_seq(train_data, batch_size, epochs)
        return model_ensemble
