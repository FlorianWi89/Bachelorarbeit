import numpy as np
import pandas as pd
import os
from tqdm import tqdm


#class to load and stack the training data

class DataLoader():
    def __init__(self, path_to_data):
        self.data_path = path_to_data
        self.file_name = "Measurements.xlsx"
        
    def load_data(self):
        
        #function to load all szenario measurements into one huge Dataframe for training
        szenarios = list(map(lambda z: int(z.replace("scenario", "")), filter(lambda z: z.startswith("scenario"), os.listdir(self.data_path))))
        
        data = pd.DataFrame()
        
        #get all files and stack them 
        for s_id in tqdm(range(len(szenarios))):
            
            pressures = pd.read_excel(os.path.join(self.data_path, f"scenario{szenarios[s_id]}", self.file_name), sheet_name="Pressures (m)", engine='openpyxl')
            pressures = pressures.drop("Timestamp", axis = 1)
            
            flows = pd.read_excel(os.path.join(self.data_path, f"scenario{szenarios[s_id]}", self.file_name), sheet_name="Flows (m3_h)", engine='openpyxl')
            flows = flows.drop("Timestamp", axis = 1)
            
            #vertical stack
            current_data = pd.concat([pressures,flows],axis=1)
            
            #horizontal stack
            data = pd.concat([data, current_data])
            
        return data
    
    def load_and_save(self):
        data = self.load_data()
        
        data = data.reset_index(drop=True)
        
        data.to_parquet('train_data.parquet.gzip',compression='gzip') 
        

#path = '/Users/florianwicher/Desktop/data/LeakageScenarios/'
#DL = DataLoader(path)


#DL.load_and_save()

