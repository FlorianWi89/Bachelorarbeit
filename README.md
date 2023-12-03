# Bachelorarbeit

## Dataloader.py
Klasse um die Trainingsdaten in ein einziges File zu packen

## FaultDetector.py
Enthält die Klassen Fault-Detector sowie Ensemble System. 
FaultDetector identifiziert die verdächtigen Punkte, EnsembleSystem wrapped die Modelle.

## FaultySensorScenario.py 
(wird nicht genutzt)

## HP_Tunings.ipynb
Experimentieren wie man die besten Hyperparameter findet.

## LinearModel.py
Wrapper Klasse für die Modelle der lower Baseline

## RNN_Plotting.ipynb
Experimentieren, wie ich Modell Vorhersagen visualisieren kann

## RecurrentNeuralNetwork.py
Wrapper Klasse für die RNNs

## Result_Inspection.ipynb
Experimentieren wie man die Ergebnisse aufbereiten kann

## eval_scenario.py
(wird nicht genutzt)

## leakage_types.json
(wird nicht genutzt)

## sensor_fault_types.json
(speichert welche Szenarien welchen Fault Type haben

## process_complete_run.py
die Main Datei die einen kompletten Run aus Training und Evaluation aller Szenarien einer Kategorie vornimmt

## save_test_cases.py
Util Funktion die alle Test Cases in ein anderes File Format speichert

## save_test_cases_leakage.py

## sort_scenarios.py
erstellt die beiden .json Dateien

## trainModel.py
Klasse, die die verschiedenen Fit Methoden aufruft je nach Modell Typ

## utils.py
Beinhatet die Eval Funktion

## train_data_ ... _parquet.gzip
