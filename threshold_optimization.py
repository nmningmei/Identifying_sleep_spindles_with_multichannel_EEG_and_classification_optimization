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

file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
lower = np.arange(0.1,1.1,0.1)
higher = np.arange(2.5,3.6,0.1)
pairs = [(l,h) for l in lower for h in higher]
result = {'sub':[],'day':[],'lower_threshold':[],'higher_threshold':[],'roc_auc':[]}

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
        
        a=Filter_based_and_thresholding()
        a.get_raw(raw)
        a.get_epochs()
        a.get_annotation(annotation)
        def cost(params):
            lower_threshold,higher_threshold = params
            a.find_onset_duration(lower_threshold,higher_threshold)
            a.sleep_stage_check()
            a.fit_predict_proba()
            a.mauanl_label()
            return metrics.roc_auc_score(a.manual_labels,a.auto_proba)
        for p in pairs:
            print(sub,day,p)
            result['sub'].append(sub)
            result['day'].append(day)
            result['lower_threshold'].append(p[0])
            result['higher_threshold'].append(p[1])
            result['roc_auc'].append(cost(p))
            
    else:
        print(sub,day,'no annotation')
result = pd.DataFrame(result)