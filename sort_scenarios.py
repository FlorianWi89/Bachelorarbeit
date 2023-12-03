import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

#1: gehe durch alle test folder und kategorisiere die scenarien nach art des fehlers

#2: lade alle scenarien in ein entsprechendes dataframe

#3: speichere die dataframes als parquet mit daten eines szenarios

sensor_fault_paths = [
    '/Users/florianwicher/Desktop/data/SensorFaultScenariosTest_0',
    '/Users/florianwicher/Desktop/data/SensorFaultScenariosTest_1',
    '/Users/florianwicher/Desktop/data/SensorFaultScenariosTest_2',
    '/Users/florianwicher/Desktop/data/SensorFaultScenariosTest_3',
    '/Users/florianwicher/Desktop/data/SensorFaultScenariosTest_4',
    '/Users/florianwicher/Desktop/data/SensorFaultScenariosTest_5',
]

leakage_paths = [
    '/Users/florianwicher/Desktop/data/LeakageScenariosTest_0',
    '/Users/florianwicher/Desktop/data/LeakageScenariosTest_1',
    '/Users/florianwicher/Desktop/data/LeakageScenariosTest_2',
    '/Users/florianwicher/Desktop/data/LeakageScenariosTest_3',
    '/Users/florianwicher/Desktop/data/LeakageScenariosTest_4',
    '/Users/florianwicher/Desktop/data/LeakageScenariosTest_5',
]

def dict_add(dict, key,value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)


def store_sensor_fault_scenarios_ordered(paths, dict):
    for i in range(len(paths)):

        folder = paths[i]

        szenarios = list(map(lambda z: int(z.replace("scenario", "")), filter(lambda z: z.startswith("scenario"), os.listdir(folder))))
        #print(f"Found {len(szenarios)} szenarios in  {folder}")
        
        for i in tqdm(range(len(szenarios))):
        
            s_id = szenarios[i]

            #### parse scenario information

            fault_info_file = list(filter(lambda z: z.endswith(".xlsx"), os.listdir(os.path.join(folder, f"scenario{s_id}", "WithoutSensorFaults"))))[0]
            df_fault_info = pd.read_excel(os.path.join(folder, f"scenario{s_id}", "WithoutSensorFaults", fault_info_file), sheet_name="Info", engine='openpyxl')

            for _, row in df_fault_info.iterrows():
                if row["Description"] == "Function type":

                    val = os.path.join(folder, f"scenario{s_id}").split('/')
                    folder_id = val[-2].split('_')[1]
                    scenario_id = val[-1].split('o')[1]
                    
                    dict_add(dict, row["Value"], {'folder_id':folder_id, 'scenario_id':scenario_id})

    for key in dict:
        print(f"Found {len(dict[key])} scenarios of type {key}")
            
    with open("sensor_fault_types.json", "w") as outfile: 
        json.dump(dict, outfile)


#=======================


def store_leakage_scenarios_ordered(paths, dict):
    for i in range(len(paths)):

        folder = paths[i]

        szenarios = list(map(lambda z: int(z.replace("scenario", "")), filter(lambda z: z.startswith("scenario"), os.listdir(folder))))
        #print(f"Found {len(szenarios)} szenarios in  {folder}")
        
        for i in tqdm(range(len(szenarios))):
        
            s_id = szenarios[i]

            #### parse scenario information

            fault_info_file = list(filter(lambda z: z.endswith(".xlsx"), os.listdir(os.path.join(folder, f"scenario{s_id}", "Leakages"))))[0]
            df_fault_info = pd.read_excel(os.path.join(folder, f"scenario{s_id}", "Leakages", fault_info_file), sheet_name="Info", engine='openpyxl')

            for _, row in df_fault_info.iterrows():
                if row["Description"] == "Leak Type":
                    dict_add(dict, row["Value"], os.path.join(folder, f"scenario{s_id}"))

    for key in dict:
        print(f"Found {len(dict[key])} scenarios of type {key}")

    with open("leakage_types.json", "w") as outfile: 
        json.dump(dict, outfile)


if __name__ == '__main__':
    
    ordered_fault_scenarios = {}
    ordered_leakage_scenarios = {}

    store_sensor_fault_scenarios_ordered(sensor_fault_paths, ordered_fault_scenarios)
    print("="*50)
    store_leakage_scenarios_ordered(leakage_paths, ordered_leakage_scenarios)


    