import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from trainModel import train_model
from utils import  eval_anomaly_detection
import json

#function to process a complete train-test run
def process_complete_train_test_run(result_path, model_type, scenario, dict):
    
    #load the training data with reduced precision
    data = pd.read_parquet(f'train_data_{scenario}.parquet.gzip').to_numpy().astype(np.float32)

    #choose the relative amount of trainig data
    data = data[: int(1 * len(data))]

    #train the whole classifier ensemble of 29 models    
    ensemble = train_model(data, 2, 29, model_type=model_type, batch_size=512, epochs=2)
    

    #iterate over every scenario type list in the json dict
    for scenario_type in dict:
        result = []
        #iterate over every test data file in the current list
        for i in tqdm(range(len(dict[scenario_type]))):

            current_test_case = dict[scenario_type][i]
            folder_id = current_test_case['folder_id']
            scenario_id = current_test_case['scenario_id']

            if scenario == 'sensor_fault':
                data_path = f'/Users/florianwicher/Desktop/TestData/SensorFaultScenariosTest_{folder_id}/{scenario_id}_{scenario}_test_data.parquet.gzip'
                info_path = f'/Users/florianwicher/Desktop/TestData/SensorFaultScenariosTest_{folder_id}/{scenario_id}_{scenario}_test_info.npz'

            if scenario == 'leakage':
                data_path = f'/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_{folder_id}/{scenario_id}_{scenario}_test_data.parquet.gzip'
                info_path = f'/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_{folder_id}/{scenario_id}_{scenario}_test_info.npz'

           
            #load model test data
            test_data = pd.read_parquet(data_path).to_numpy().astype(np.float32)

            #load metadata  
            labels = np.load(info_path, allow_pickle=True)['y']
    
            #====== Anomaly detection ======#

            #let the models predict the sensor data and get suspicious points
            suspicious_time_points = ensemble.apply_detector(test_data)
            
            faults_time = np.where(labels == 1)[0]
    
            # Remove all false alarms
            suspicious_time_points = list(filter(lambda t: t in faults_time, suspicious_time_points))

            #get classification scores
            res = eval_anomaly_detection(suspicious_time_points, faults_time, labels)
            res["scenario"] = f'{folder_id}_{scenario_id}'
        
        result.append(res)

        if scenario == 'sensor_fault':
            result_path = os.path.join(f'/Users/florianwicher/Desktop/resultData/Sensor_fault', f'{model_type}_{scenario}_{scenario_type}_result.parquet.gzip')
        
        if scenario == 'leakage':
            result_path = os.path.join(f'/Users/florianwicher/Desktop/resultData/Leakage', f'{model_type}_{scenario}_{scenario_type}_result.parquet.gzip')
        
        result = pd.DataFrame(result).to_parquet(result_path, compression='gzip')

if __name__ == "__main__":

    #first comand line argument ist the model that should be trained and evaluated
    model_type = sys.argv[1]

    #the second param is the scenario : Leakage or sensor_fault
    scenario = sys.argv[2]
    
    result_path = f'/Users/florianwicher/Desktop/resultData/Sensor_fault'

    with open(f'{scenario}_types.json') as json_file:
        data = json.load(json_file)

    process_complete_train_test_run(result_path, model_type, scenario, data)






