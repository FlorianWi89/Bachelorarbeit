import pandas as pd
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

##############################################################


class BaseRNN(nn.Module):
    def __init__(self):
        super(BaseRNN, self).__init__()

    def forward(self, x):
        pass

    def fit(self, X, y, batch_size=128, epochs=1, lr=0.01, val_split=0.2):
        
        num_train = int((1-val_split) * len(X))
        X_train = X[: num_train]
        y_train = y[: num_train]

        X_test  = X[num_train: ]
        y_test  = y[num_train:]
        
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
            #print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Duration: {round((time.time() - start_time)/60, 3)}')
            train_loss_vals.append(round(loss.item(),4))
            
            val_loss = 0.0
            self.eval()
            
            with torch.no_grad():
                
                for batch_X, batch_y in test_loader:
                    pred = self.forward(batch_X)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / batch_size
            test_loss_vals.append(float(val_loss))
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss} , Duration: {round((time.time() - start_time)/60, 3)}')
            
        return (train_loss_vals, test_loss_vals)
    
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

class RNN(BaseRNN):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.tanh = nn.tanh()
        self.linear2 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout(out)
        out, _ = self.rnn2(out)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out
    
class GRU(BaseRNN):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(30, 30, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.gru2 = nn.GRU(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.linear2 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout(out)
        out, _ = self.gru2(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

class LSTM(BaseRNN):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(30, 30, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(30, 30, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(30, 15)
        self.linear2 = nn.Linear(15, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out