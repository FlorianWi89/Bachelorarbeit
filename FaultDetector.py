import numpy as np
from joblib import Parallel, delayed


#class for predicting time series and identifying suspicious time points
class FaultDetector():
    def __init__(self, model , threshold = 0.5):
        self.model = model
        self.threshold = threshold
        
    def get_suspicious_time_points(self, y_pred, y_true):
        
        #get the points where the resudual is bigger that the given threshold
        result = list(np.where(np.abs(y_pred-y_true) > self.threshold)[0])
        
        return result


#wrapper class for a model and its according fault detector
class EnsembleSystem():
    def __init__(self, model_class, flow_nodes, pressure_nodes, model_type ,fault_detector = FaultDetector):
        self.model_class = model_class          #wrapper class of the model
        self.flow_nodes = flow_nodes            #number of flow nodes in the network
        self.pressure_nodes = pressure_nodes    #number of pressure nodes in the network
        self.model_type = model_type            #model type
        
        self.fault_detector = fault_detector    #the trained fault detector
        self.models = []
    
    
    

    #function to fit all  models of the ensemble 
    def fit_seq(self, X_train, batch_size=0, epochs=0):

        #indices is a list with indices for the models
        for pressure_node in range(self.pressure_nodes):
            
            #get the feature indices and the target index
            inputs_idx = list(range(X_train.shape[1]));inputs_idx.remove(pressure_node)
            
            x_train, y_train = X_train[:,inputs_idx], X_train[:,pressure_node]
    
            #Create model
            model = self.model_class()

            #fit model based on the given model type
            if self.model_type in ['LSTM', 'GRU', 'RNN']:

                model.fit(x_train, y_train, batch_size, epochs)
            else:
                model.fit(x_train, y_train)
            
          
            
            #build fault detector   
            
            max_error, mean_error, min_error = model.get_max_mean_min_prediction_error(x_train, y_train)
            #print(f"Model {pressure_node} trained -- Mean Error: ",1.2 * mean_error)
            print(f"Fitted Model {pressure_node}")
            print("#"*40)
            fault_detector = self.fault_detector(model, 1.2 * mean_error)
            
            #store model
            self.models.append({"model": model, "fault_detector": fault_detector, "input_idx": inputs_idx, "target_idx": pressure_node})
    
        
            
    
    #function to invoke the models predict function
    def predict(self, X):
        
        Y = []
        
        for model in self.models:
            pred = model['model'].predict(X[:, model['input_idx']])
            Y.append(pred)
            
        return np.array(Y).flatten()
    

    def score(self, X, y_true, model_id=1):
        pass
        
    
    #function to apply the detector on new test data
    def apply_detector(self, X):
        
        suspicious_time_points = []        
        
        for model in self.models:
            x = X[:, model['input_idx']]
            y_true = X[:, model['target_idx']]

            #let the model make a prediction on the new X data
            y_pred = model["model"].predict(x).flatten()
            #print(y_pred)
            
            #build the residuals between y_true and y_pred
            suspicious_time_points += model["fault_detector"].get_suspicious_time_points(y_pred, y_true)
            
        suspicious_time_points = list(set(suspicious_time_points));suspicious_time_points.sort()
        #print(f"Inference Done, Sus Time Points: {suspicious_time_points}")
        return suspicious_time_points
        
        

