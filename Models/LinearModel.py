from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

class LinearModel():
    def __init__(self, model_type = 'LinearRegression'):
        self.model_type = model_type
        
        if model_type == 'LinearRegression':
            self.model = LinearRegression()
        elif model_type == 'Ridge':
            self.model = Ridge()
        
    def fit(self, X, y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def get_max_mean_min_prediction_error(self, X, y):
        
        y_pred = self.model.predict(X)

        errors_in_prediction = np.abs(y_pred - y)

        return np.max(errors_in_prediction), np.mean(errors_in_prediction), np.min(errors_in_prediction)