import numpy as np

def get_train_test_split(X,y, train_size = 5000, network_units=31):
    X_len = len(X)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = np.array(X_train).reshape((train_size, network_units,1))
    X_test = np.array(X_test).reshape((X_len - train_size, network_units,1))

    y_train = np.array(y_train).reshape((train_size,1))
    y_test = np.array(y_test).reshape((X_len - train_size,1))

    
    return X_train, X_test, y_train, y_test


def get_possible_fault_points(prediction, ground_truth, train_len, threshold = 0.5):
    
    pred = np.array(prediction)
    true = np.array(ground_truth).flatten()
    differences = np.abs( pred - true)
    
    fault_indices = np.where(differences > threshold)[0]
    
    #get real index by adding the length of the train sequence
    fault_indices = fault_indices + train_len
    
    return fault_indices



def evaluate_prediction(fault_start, fault_end, prediction, ground_truth, train_size, fault_threshold):
    possible_fault_points = get_possible_fault_points(prediction, ground_truth, train_size, fault_threshold)

    total_faults = np.arange(fault_start,fault_end+1)

    correct_predictions = len(np.intersect1d(np.arange(fault_start, fault_end+1), possible_fault_points))

    #number of correct classified fault_days
    TPR = round((correct_predictions / len(total_faults)) * 100, 2)

    #number of incorrect classified fault_days
    FPR = round(((len(total_faults)-correct_predictions) / correct_predictions) * 100 , 2)

    return TPR, FPR


def evaluate_fault_detection(faults_time_pred, faults_time_truth, tol=10, false_alarms_tol=2, use_intervals=True):
    false_alarms = 0
    faults_detected = 0
    faults_not_detected = 0

    # Extract intervals in which a fault is present
    intervals = []
    i = 0
    t0 = faults_time_truth[i]
    while i < len(faults_time_truth)-1:
        if faults_time_truth[i + 1] != faults_time_truth[i] + 1:
            intervals.append((t0, faults_time_truth[i]))
            t0 = faults_time_truth[i + 1]
        if not (i + 1 < len(faults_time_truth)-1):
            intervals.append((t0, faults_time_truth[i+1]))

        i += 1

    # Check for false alarms
    for i in range(len(faults_time_pred)):
        t = faults_time_pred[i]
        b = False
        for dt in faults_time_truth:
            if dt - tol <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            if i + false_alarms_tol <= len(faults_time_pred)-1:    # Need a minimum of number of continous alarms for triggering a "real alarm" -- ignore noise!
                if all([t + j == faults_time_pred[i+j] for j in range(false_alarms_tol)]):
                    false_alarms += 1
    
    # Check for detected and undetected faults
    if use_intervals:
        for t0, t1 in intervals:
            b = False
            for t in faults_time_pred:
                if t0 <= t and t <= t1: # TODO: Use tolerance?
                    b = True
                    faults_detected += 1
                    break

            if b is False:
                faults_not_detected += 1
    else:
        for dt in faults_time_truth:
            b = False
            for t in faults_time_pred:
                if dt - tol <= t and t <= dt + tol:
                    b = True
                    faults_detected += 1
                    break
            if b is False:
                faults_not_detected += 1

    return {"false_positives": false_alarms, "true_positives": faults_detected, "false_negatives": faults_not_detected}



def eval_anomaly_detection(suspicious_time_points, faults_time, fault_labels_all_test):
    test_times = range(len(fault_labels_all_test))
    test_minus_pred = list(set(test_times) - set(suspicious_time_points))

    # Compute detection delay
    try: 
        dd = list(faults_time).index(suspicious_time_points[0])
    #detection_delay = 1
    except:
        dd = -1

    # Compute TPs, FPs, etc. for every point in time when a fault is present
    TP = np.sum([t in faults_time for t in suspicious_time_points]) / len(suspicious_time_points)
    FP = np.sum([t not in faults_time for t in suspicious_time_points]) / len(suspicious_time_points)
    FN = np.sum([t in faults_time for t in test_minus_pred]) / len(test_minus_pred)
    TN = np.sum([t not in faults_time for t in test_minus_pred]) / len(test_minus_pred)

    # Export results
    return {"detection_delay": dd, "tp": TP, "fp": FP, "fn": FN, "tn": TN}