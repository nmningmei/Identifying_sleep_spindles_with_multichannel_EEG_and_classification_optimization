# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:21:28 2017

@author: ning
"""

from utilities import *
from Filter_based_and_thresholding import Filter_based_and_thresholding
import eegPinelineDesign
import numpy as np
import mne
import pandas as pd
from sklearn import metrics
import os
from random import shuffle
from collections import Counter

file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
lower = np.arange(0.1,1.1,0.1)
higher = np.arange(2.5,3.6,0.1)
pairs = [(l,h) for l in lower for h in higher]
saving_dir = 'D:\\NING - spindle\\training set\\re-run\\'
if not os.path.exists(saving_dir):
    #print(directory_2)
    os.makedirs(saving_dir)
result = {'sub':[],'day':[],'lower_threshold':[],'higher_threshold':[],'roc_auc':[]}
shuffle(list_file_to_read)
for file in list_file_to_read:
    sub = file.split('_')[0]
    if int(sub[3:]) >= 11:
        day = file.split('_')[2][:4]
        old = False
    else:
        day = file.split('_')[1]
        old = True

    annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
    if len(annotation_file) != 0:
        print('gathering...................')
        annotation = pd.read_csv(annotation_file[0])
        ################### eeg data part ###########################
        raw = mne.io.read_raw_fif(file,preload=True)
        if old:
            pass
        else:
            raw.resample(500, npad="auto") # down sampling Karen's data
        
        a=Filter_based_and_thresholding(moving_window_size=500)
        a.get_raw(raw)
        a.get_epochs()
        a.get_annotation(annotation)
        a.mauanl_label()
        if a.spindles.shape[0] < 40:
            print(sub,day,'pass')
            pass
        else:
            def cost(params):
                lower_threshold,higher_threshold = params
                print('FBT model')
                a.find_onset_duration(lower_threshold,higher_threshold)
                print('sleep stage 2 check')
                a.sleep_stage_check()
                print('compute probability')
                a.prepare_validation()
                try:
                    a.fit()
                    return metrics.roc_auc_score(a.manual_labels,a.auto_proba)
                except:
                    print('unable to find spindles with these pairs of thresholds')
                    print(Counter(a.auto_label))
            for p in pairs:
                try:
                    print(sub,day,p)
                    result['roc_auc'].append(cost(p))
                    result['sub'].append(sub)
                    result['day'].append(day)
                    result['lower_threshold'].append(p[0])
                    result['higher_threshold'].append(p[1])
                    
                    temp = pd.DataFrame(result)
                    temp.to_csv(saving_dir+'temp_result.csv',index=False)
                except:
                    print(sub,day,p,"I am tired of debugging")
            
    else:
        print(sub,day,'no annotation')
result = pd.DataFrame(result)