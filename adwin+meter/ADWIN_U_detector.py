from timeit import default_timer as timer
from skmultiflow.drift_detection.adwin import ADWIN
import os
import pandas as pd
import numpy as np
from random import seed, shuffle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

def compute_BAR(acc, requested_labels):
    BAR = 2 * ((acc * (100 - requested_labels)) / (acc + (100 - requested_labels)))
    return BAR

def ADWIN_U(TRAIN_FILENAME, TEST_FILENAME, statistic, model_dependent): 
    train_data = pd.read_csv(TRAIN_FILENAME, header=None, index_col=False,sep=',')
    test_data = pd.read_csv(TEST_FILENAME, header=None, index_col=False,sep=',')

    train_X = train_data.iloc[:,:-1]
    test_X = test_data.iloc[:,:-1]
    train_y = train_data.iloc[:,-1]
    test_y = test_data.iloc[:,-1]    
    
    if model_dependent:        
        alldata = pd.concat([train_y, test_y], ignore_index=True)        
        classes = np.unique(alldata)
        detector = {}    
        for c in classes:
            detector[c] = ADWIN()
    else:
        detector = ADWIN()
    
    #print('ADWIN-U (Model dependent) Running...')    
    model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)    
    model.fit(train_X, train_y)


    vet_acc = np.zeros(len(test_y))
    drift_points = []
    requested_labels = [0]
    start = timer()
    
    for i in range(0, len(test_X)):  
        #print('Example {}/{}'.format(i+1, len(test_y)),end='\r')
        prediction = model.predict(test_X.iloc[[i]]) 
        prediction = prediction[0]        
        if prediction == test_y[i]:
            vet_acc[i] = 1
         
        
        # extract statistic from the i-th stream example
        if statistic == 'mean':
            stat = np.mean(test_X.iloc[i])
        elif statistic == 'median':
            stat = np.median(test_X.iloc[i])
        elif statistic == 'variance':
            stat = np.var(test_X.iloc[i])
        elif statistic == 'harmonic mean':        
            #stat = stats.hmean(np.abs(test_X.iloc[i]), nan_policy='omit')
            stat = stats.hmean(np.abs(test_X.iloc[i]))
        elif statistic == 'geometric mean':            
            #stat = stats.gmean(np.abs(test_X.iloc[i]), nan_policy='omit')
            stat = stats.gmean(np.abs(test_X.iloc[i]))
        elif statistic == 'std':            
            stat = np.std(test_X.iloc[i])
        elif statistic == 'skewness':        
            stat = stats.skew(test_X.iloc[i], bias=False)
        elif statistic == 'kurtosis':            
            stat = stats.kurtosis(test_X.iloc[i], bias=False)
        elif statistic == 'variation':            
            #stat = stats.variation(test_X.iloc[i], nan_policy='omit')
            stat = stats.variation(test_X.iloc[i])
        elif statistic == 'mad':            
            stat = stats.median_abs_deviation(test_X.iloc[i])
        elif statistic == 'probability':   
            probabilities = model.predict_proba(test_X.iloc[[i]])            
            stat = np.max(probabilities)
        elif statistic == "skewness-kurtosis":
            stat1 = stats.skew(test_X.iloc[i], bias=False)
            stat2 = stats.kurtosis(test_X.iloc[i], bias=False)
            stat = stat1/stat2


            
        if model_dependent:          
            detector[prediction].add_element(stat)
            min_window = np.Inf
            flag = False
            for c in classes:
                if detector[c].detected_change():
                    flag = True
                    w = detector[c].width
                    if w < min_window:
                        min_window = w
                    detector[c].reset() 
            if flag:
                drift_points.append(i) 
                model.fit(test_X[i-min_window:i].reset_index(drop=True), test_y[i-min_window:i])  
                requested_labels.append(requested_labels[-1]+min_window) 
            else:
                requested_labels.append(requested_labels[-1])                
            
        else:
            detector.add_element(stat)
            if detector.detected_change():
                w = detector.width
                detector.reset()
                drift_points.append(i)
                model.fit(test_X[i-w:i].reset_index(drop=True), test_y[i-w:i])   
                requested_labels.append(requested_labels[-1]+w)
            else:
                requested_labels.append(requested_labels[-1])
                   

    end = timer()
    execution_time = end-start 
    mean_acc = np.mean(vet_acc)*100
    rl = (requested_labels[-1] * 100)/len(test_y)
    
    BAR = compute_BAR(mean_acc, rl)
    '''
    print('\nFinished!')	
    print('{} drifts detected at {}'.format(len(drift_points), drift_points))
    print('Average classification accuracy: {}%'.format(np.round(mean_acc,2)))
    print('BAR: {}%'.format(np.round(BAR,2)))
    print('Time per example: {} sec'.format(np.round(execution_time/len(test_y),2)))
    print('Total time: {} sec'.format(np.round(execution_time,2)))
    plot_acc(vet_acc, 500, '', 'dashed', 'ADWIN-U')
    plot_requested_labels(requested_labels, drift_points, vet_acc, '', 'dashed', 'ADWIN-U')
    '''        
    return (drift_points, requested_labels[1:], vet_acc, mean_acc, BAR, execution_time)