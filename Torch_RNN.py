import torch
import time
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class Torch_RNN(nn.Module):
    def __init__(self):
        super(Torch_RNN, self).__init__()
        self.lstm1 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.23)
        self.lstm2 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(15, 15)
        self.linear3 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        return out
    
    def fit(self, X, y, batch_size=128, epochs=1):

        self.train()
        # Konvertieren der Daten in PyTorch-Tensoren
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Erstellen von DataLoader-Objekten für das effiziente Laden von Daten während des Trainings
        batch_size = 512
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        # Definition von Verlust und Optimierer
        criterion = nn.L1Loss()  # Mean Absolute Error
        optimizer = torch.optim.Adam(self.parameters())

        for epoch in range(epochs):
            start_time = time.time()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Ausgabe des Verlusts pro Epoche
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Duration: {round((time.time() - start_time)/60, 3)}')


    #let the models predict and return a reshaped output
    def predict(self, X):
        
        X_test = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            test_outputs = self.forward(X_test)
        
        return test_outputs.flatten()
    
    #return some stats for the predictoin errors
    def get_max_mean_min_prediction_error(self, X, y):
        
        errors_in_prediction = np.abs(np.subtract(np.array(self.predict(X)), np.array(y)))

        return np.max(errors_in_prediction), np.mean(errors_in_prediction), np.min(errors_in_prediction)