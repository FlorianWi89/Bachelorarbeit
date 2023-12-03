import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from trainModel import train_model
from utils import  eval_anomaly_detection


#function to process a complete train-test run
def process_complete_train_test_run(test_data_paths, test_data_files, result_path, model_type):
    
    #load the training data with reduced precision
    data = pd.read_parquet('train_data_leakage.parquet.gzip').to_numpy().astype(np.float32)

    #choose the relative amount of trainig data
    data = data[: int(0.2 * len(data))]

    #train the whole classifier ensemble of 29 models    
    ensemble = train_model(data, 2, 29, model_type=model_type, batch_size=128, epochs=2)
    
    result = []

    #iterate over every test data folder
    for test_folder in test_data_paths:

        test_files = test_data_files[0]

        data_path = test_folder

        folder_id = data_path[-1]

        #iterate over every test data file in the current folder
        for i in tqdm(range(len(test_files))):
            
            test_data = os.path.join(data_path, test_files[i][0])

            test_data_info = os.path.join(data_path, test_files[i][1])
    
            scenario_id = test_data.split('/')[-1].split('_')[0]
    
            #load model test data
            test_data = pd.read_parquet(test_data).to_numpy()

            #load metadata  
            labels = np.load(test_data_info, allow_pickle=True)['y']
    
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
        
    result_path = os.path.join(result_path, f'{model_type}_result.parquet.gzip')
        
    result = pd.DataFrame(result).to_parquet(result_path, compression='gzip')



    
if __name__ == "__main__":
    
    test_data_path_0 = '/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_0'
    test_data_path_1 = '/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_1'
    test_data_path_2 = '/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_2'
    test_data_path_3 = '/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_3'
    test_data_path_4 = '/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_4'
    test_data_path_5 = '/Users/florianwicher/Desktop/TestData/LeakageScenariosTest_5'

    test_data_paths = [test_data_path_0,test_data_path_1,test_data_path_2,test_data_path_3,test_data_path_4,test_data_path_5]

    
    result_path = '/Users/florianwicher/Desktop/resultData/Leakage'

    ####

    test_data_files_0 = list(filter(lambda z: z.endswith("test_data.parquet.gzip"), os.listdir(test_data_path_0)))
    test_info_files_0 = list(filter(lambda z: z.endswith("test_info.npz"), os.listdir(test_data_path_0)))

    test_data_files_0.sort()
    test_info_files_0.sort()

    ####

    test_data_files_1 = list(filter(lambda z: z.endswith("test_data.parquet.gzip"), os.listdir(test_data_path_1)))
    test_info_files_1 = list(filter(lambda z: z.endswith("test_info.npz"), os.listdir(test_data_path_1)))

    test_data_files_1.sort()
    test_info_files_1.sort()

    ####

    test_data_files_2 = list(filter(lambda z: z.endswith("test_data.parquet.gzip"), os.listdir(test_data_path_2)))
    test_info_files_2 = list(filter(lambda z: z.endswith("test_info.npz"), os.listdir(test_data_path_2)))

    test_data_files_2.sort()
    test_info_files_2.sort()

    ####

    test_data_files_3 = list(filter(lambda z: z.endswith("test_data.parquet.gzip"), os.listdir(test_data_path_3)))
    test_info_files_3 = list(filter(lambda z: z.endswith("test_info.npz"), os.listdir(test_data_path_3)))

    test_data_files_3.sort()
    test_info_files_3.sort()

    ####

    test_data_files_4 = list(filter(lambda z: z.endswith("test_data.parquet.gzip"), os.listdir(test_data_path_4)))
    test_info_files_4 = list(filter(lambda z: z.endswith("test_info.npz"), os.listdir(test_data_path_4)))

    test_data_files_4.sort()
    test_info_files_4.sort()

    ####

    test_data_files_5 = list(filter(lambda z: z.endswith("test_data.parquet.gzip"), os.listdir(test_data_path_5)))
    test_info_files_5 = list(filter(lambda z: z.endswith("test_info.npz"), os.listdir(test_data_path_5)))

    test_data_files_5.sort()
    test_info_files_5.sort()

    ####

    model='Simple-RNN'
    
    test_files_0 = list(zip(test_data_files_0, test_info_files_0))  
    test_files_1 = list(zip(test_data_files_1, test_info_files_1))
    test_files_2 = list(zip(test_data_files_2, test_info_files_2))
    test_files_3 = list(zip(test_data_files_3, test_info_files_3))
    test_files_4 = list(zip(test_data_files_4, test_info_files_4))
    test_files_5 = list(zip(test_data_files_5, test_info_files_5))

    test_data_files = [test_files_0,test_files_1,test_files_2,test_files_3,test_files_4,test_files_5]

    process_complete_train_test_run(test_data_paths, test_data_files, result_path, model)
   

