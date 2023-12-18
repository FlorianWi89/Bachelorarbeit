import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
pd.reset_option("mode.chained_assignment")


#code to save all test case excel files as parquet

data_path = '/Users/florianwicher/Desktop/data/'

test_folders = list(filter(lambda z: z.startswith("SensorFaultScenariosTest"), os.listdir(data_path)))


#for every folder load the files and save them as parquet

out_path = '/Users/florianwicher/Desktop/TestData'



def save_test_cases(data_path ,output_path):

    szenarios = list(map(lambda z: int(z.replace("scenario", "")), filter(lambda z: z.startswith("scenario"), os.listdir(data_path))))

    os.mkdir(output_path)

    os.chdir(output_path)

    for i in tqdm(range(len(szenarios))):
        
        s_id = szenarios[i]

        #### parse scenario information

        fault_info_file = list(filter(lambda z: z.endswith(".xlsx"), os.listdir(os.path.join(data_path, f"scenario{s_id}", "WithoutSensorFaults"))))[0]
        df_fault_info = pd.read_excel(os.path.join(data_path, f"scenario{s_id}", "WithoutSensorFaults", fault_info_file), sheet_name="Info", engine='openpyxl')

        for _, row in df_fault_info.iterrows():
            if row["Description"] == "Fault Start":
                faulty_sensor_start = row["Value"]

            elif row["Description"] == "Fault End":
                faulty_sensor_end = row["Value"]

        ####
        
        szenario_path = os.path.join(data_path, f"scenario{s_id}", 'Measurements.xlsx')
        
        pressures = pd.read_excel(szenario_path, sheet_name="Pressures (m)", engine='openpyxl')
        flows = pd.read_excel(szenario_path, sheet_name="Flows (m3_h)", engine='openpyxl')

        # Create labels
        df_labels = pressures[["Timestamp"]].copy()
        df_labels["label"] = 0

        indices = df_labels[(df_labels["Timestamp"] >= faulty_sensor_start) & (df_labels["Timestamp"] <= faulty_sensor_end)].index
        #print("Indices: ", indices)
        for idx in indices:
            df_labels["label"].loc[idx] = 1

        labels = df_labels["label"].to_numpy().flatten()

        pressures = pressures.drop("Timestamp", axis = 1)
        flows = flows.drop("Timestamp", axis = 1)

        #vertical stack
        data = pd.concat([pressures,flows],axis=1)
        data = data.reset_index(drop=True)

        data.to_parquet(os.path.join(output_path, f"{s_id}_sensor_fault_test_data.parquet.gzip"), compression='gzip')
        np.savez(os.path.join(output_path, f"{s_id}_sensor_fault_test_info"),y = labels)
        


if __name__ == "__main__":

    #code to save all test case excel files as parquet

    data_path = '/Users/florianwicher/Desktop/data/'

    test_folders = list(filter(lambda z: z.startswith("SensorFaultScenariosTest"), os.listdir(data_path)))


    #for every folder load the files and save them as parquet

    out_path = '/Users/florianwicher/Desktop/TestData'

    data_paths = [os.path.join(data_path, folder) for folder in test_folders]
    out_paths = [os.path.join(out_path, folder) for folder in test_folders]

    paths = zip(data_paths, out_paths)

    results = Parallel(n_jobs=-2)(delayed(save_test_cases)(e[0], e[1]) for e in paths)
