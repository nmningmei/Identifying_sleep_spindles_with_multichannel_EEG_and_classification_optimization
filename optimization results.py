# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:03:52 2017

@author: ning
"""

import os
import pandas as pd
import numpy as np
import mne
import seaborn as sns
from Filter_based_and_thresholding import Filter_based_and_thresholding
import re
def stage_2(x):
    key1 = re.compile('markon: 2',re.IGNORECASE)
    key2 = re.compile('markon:2',re.IGNORECASE)
    key3 = re.compile('Markon:2',re.IGNORECASE)
    if (key1.search(x) is not None) or (key2.search(x) is not None) or (key3.search(x) is not None):
        return True
    else:
        return False
def get_median_peak_to_peak(model):
    peak_to_peak = []
    power = []
    for ii,(_,(duration,onset)) in enumerate(model.auto_scores[['Duration','Onset']].iterrows()):
        start = onset - duration/2
        stop = onset + duration/2
        start, stop = raw.time_as_index([start,stop])
        segment, time = raw[:,start:stop]
        p,_ = mne.time_frequency.psd_array_multitaper(segment,500,fmin=11,fmax=16)
        p = p.mean(1)
        power.append(10*np.log10(p))
        
        temp = []
        for jj,ch_name in enumerate(raw.ch_names):
            EvokeArray = mne.EvokedArray(segment,raw.info,)
            EvokeArray.pick_channels([ch_name])
            _,pos_time_index = EvokeArray.get_peak(time_as_index=True,mode='pos')
            _,neg_time_index = EvokeArray.get_peak(time_as_index=True,mode='neg')
            temp.append(segment[jj][pos_time_index] - segment[jj][neg_time_index])
        peak_to_peak.append(np.median(temp))
        
        
    return np.array(peak_to_peak),np.array(power)
os.chdir('D:\\NING - spindle\\training set\\re-run\\')
df = pd.read_csv('temp_result.csv')
df = df.dropna()
orders = np.concatenate([np.sort(pd.unique(df['sub']))[-4:],np.sort(pd.unique(df['sub']))[:-4]])
optimized = []
for sub in orders:
    for day in [1,2]:
        try:
            work = df[(df['sub']==sub) & ((df['day'] == 'day'+str(day)) | (df['day'] == 'd'+str(day)))]
            print(work.loc[work['roc_auc'].idxmax()])
            optimized.append(work.loc[work['roc_auc'].idxmax()])
        except:
            print('might not have a different day')
            
optimized = pd.DataFrame(optimized).reset_index()
optimized = optimized[optimized.columns[1:]]
optimized.to_csv('optimzed results.csv',index=False)
thresholds1=optimized.iloc[:7][['higher_threshold','lower_threshold']].median().values # forest's thresholds - median
thresholds2=optimized.iloc[:7][['higher_threshold','lower_threshold']].mean().values# forest's thresholds - mean
thresholds3=optimized[['higher_threshold','lower_threshold']].median().values # both scorers - median
os.chdir('D:\\NING - spindle\\training set\\')
behavioral = pd.read_excel('spindle result_behavioral data_updated October 4 2016.xlsx')
if True:
    results = {'sub':[],
               'load':[],
               'spindle':[],
               'stage_2_time':[],
               'density':[],
               'WM_dprime':[],
               'Rec1_dprime':[],
               'Rec2_dprime':[],
               'higher_threshold':[],
               'lower_threshold':[],
               'median_peak_to_peak':[],
               'median_power_density':[]}
    for Ts in [thresholds1,thresholds2,thresholds3]:
        for sub_n in np.arange(11,33):
            annotation_files = [f for f in os.listdir('D:\\NING - spindle\\training set\\') if (str(sub_n) in f) and ('annotation' in f) and ('nap' in f)]
            print(annotation_files)
            if len(annotation_files) > 0:
                for anno in annotation_files:
                    annotation = pd.read_csv(anno)
                    try:
                        sub,day,_ = anno.split('_')
                        day = day[:4]
                    except:
                        sub,_,day,_,_=anno.split('_')
                    raw_ = [f for f in os.listdir('D:\\NING - spindle\\training set\\') if (sub in f) and ('fif' in f) and (day in f)][0]
                    raw = mne.io.read_raw_fif(raw_,preload=True)
                    old = False if int(sub[3:]) > 10 else True
                    if not old:
                        raw.resample(500)
                    _,load,_ = raw_.split('_');load = int(load[1])
                    model = Filter_based_and_thresholding()
                    model.get_raw(raw)
                    model.get_annotation(annotation)
                    higher, lower = Ts
                    model.find_onset_duration(lower,higher)
                    model.sleep_stage_check()
                    n_spindle = model.auto_scores.shape[0]
                    stage_2_time = np.sum(annotation['Annotation'].apply(stage_2)) * 30
                    spindle_density = n_spindle/stage_2_time * 60
                    WM = behavioral[(behavioral['SubjNum']==int(sub[3:])) & (behavioral['Condition']==load)]['WM_dprime'].values[0]
                    Rec1 = behavioral[(behavioral['SubjNum']==int(sub[3:])) & (behavioral['Condition']==load)]['Rec1_dprime'].values[0]
                    Rec2 = behavioral[(behavioral['SubjNum']==int(sub[3:])) & (behavioral['Condition']==load)]['Rec2_dprime'].values[0]
                    peak_to_peak, power = get_median_peak_to_peak(model)
                    peak_to_peak = np.median(peak_to_peak)
                    power = np.median(power)
                    results['sub'].append(int(sub[3:]))
                    results['load'].append(load)
                    results['spindle'].append(n_spindle)
                    results['stage_2_time'].append(stage_2_time)
                    results['density'].append(n_spindle/stage_2_time * 60)
                    results['WM_dprime'].append(WM)
                    results['Rec1_dprime'].append(Rec1)
                    results['Rec2_dprime'].append(Rec2)
                    results['higher_threshold'].append(higher)
                    results['lower_threshold'].append(lower)
                    results['median_peak_to_peak'].append(peak_to_peak)
                    results['median_power_density'].append(power)
                    
    results = pd.DataFrame(results)
    results.to_csv('D:\\NING - spindle\\training set\\re-run\\more_measures.csv',index=False)
results = pd.read_csv('D:\\NING - spindle\\training set\\re-run\\more_measures.csv')

results = results.replace(np.inf,0)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(16,16),nrows=3,ncols=3)
for Ts,row_axes in zip([thresholds1,thresholds2,thresholds3],axes):
    higher, lower = Ts
    working = results[(results['higher_threshold'] == higher) & (results['lower_threshold'] == lower)]
    for predictee,ax in zip(['WM_dprime','Rec1_dprime','Rec2_dprime'],row_axes):
        ax = sns.regplot(x='density',y=predictee,data=working[working['load']==2],robust=False,color='blue',marker='+',ax=ax,label='load 2')
        ax = sns.regplot(x='density',y=predictee,data=working[working['load']==5],robust=False,color='red',marker='o',ax=ax,label='load 5')
        ax.set(title='higher:%.2f,lower:%.2f'%(higher,lower),xlabel='density',ylabel=predictee)
        ax.legend(loc='upper right')
fig.savefig('D:\\NING - spindle\\training set\\re-run\\compare across thrseholds (density).png',dpi=500)   
fig, axes = plt.subplots(figsize=(16,16),nrows=3,ncols=3)
for Ts,row_axes in zip([thresholds1,thresholds2,thresholds3],axes):
    higher, lower = Ts
    working = results[(results['higher_threshold'] == higher) & (results['lower_threshold'] == lower)]
    for predictee,ax in zip(['WM_dprime','Rec1_dprime','Rec2_dprime'],row_axes):
        ax = sns.regplot(x='median_peak_to_peak',y=predictee,data=working[working['load']==2],robust=False,color='blue',marker='+',ax=ax,label='load 2')
        ax = sns.regplot(x='median_peak_to_peak',y=predictee,data=working[working['load']==5],robust=False,color='red',marker='o',ax=ax,label='load 5')
        ax.set(title='higher:%.2f,lower:%.2f'%(higher,lower),xlabel='median peak to peak amplitude ($\mu$V)',ylabel=predictee)
        ax.legend(loc='upper right')
fig.savefig('D:\\NING - spindle\\training set\\re-run\\compare across thrseholds (median_peak_to_peak).png',dpi=500)    
fig, axes = plt.subplots(figsize=(16,16),nrows=3,ncols=3)
for Ts,row_axes in zip([thresholds1,thresholds2,thresholds3],axes):
    higher, lower = Ts
    working = results[(results['higher_threshold'] == higher) & (results['lower_threshold'] == lower)]
    for predictee,ax in zip(['WM_dprime','Rec1_dprime','Rec2_dprime'],row_axes):
        ax = sns.regplot(x='median_power_density',y=predictee,data=working[working['load']==2],robust=False,color='blue',marker='+',ax=ax,label='load 2')
        ax = sns.regplot(x='median_power_density',y=predictee,data=working[working['load']==5],robust=False,color='red',marker='o',ax=ax,label='load 5')
        ax.set(title='higher:%.2f,lower:%.2f'%(higher,lower),xlabel='median power density',ylabel=predictee)
        ax.legend(loc='upper right')
fig.savefig('D:\\NING - spindle\\training set\\re-run\\compare across thrseholds (median_power_density).png',dpi=500)          


