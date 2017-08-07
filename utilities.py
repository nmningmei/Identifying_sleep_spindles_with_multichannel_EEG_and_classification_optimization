# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:51:07 2017

@author: ning
"""
import numpy as np

def window_rms(segment,window_size):
    from scipy import signal
    segment_squre = np.power(segment,2)
    window = signal.gaussian(window_size,(window_size/.68)/2)
    return np.sqrt(np.convolve(segment_squre,window,'same')/len(segment))*1e2
def trimmed_std(a,p=0.05):
    from scipy import stats
    temp = stats.trimboth(a,p/2)
    return np.std(temp)
def stage_check(x):
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
def intervalCheck(a,b,tol=0):#a is an array and b is a point
    return a[0]-tol <= b <= a[1]+tol
def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) < min(x2,y2)
def psuedo_rms(lower_threshold, higher_threshold,signal,sample_size=500):
    from scipy.stats import trim_mean
    rms = window_rms(signal,sample_size)
    l = trim_mean(rms,0.05) + lower_threshold * trimmed_std(rms,0.05)
    h = trim_mean(rms,0.05) + higher_threshold* trimmed_std(rms,0.05)
    prop = (sum(rms>l)+sum(rms<h))/(sum(rms<h) - sum(rms<l))
    if np.isinf(prop):
        prop = (sum(rms>l)+sum(rms<h))
    return prop
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    if spindle_duration_fix: # a manually marked spindle
        spindle_start = spindle - 0.5
        spindle_end   = spindle + 1.5
        return is_overlapping(time_interval[0],time_interval[1],
                              spindle_start,spindle_end)
    else: # an automated marked spindle, which was marked at its peak
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        return is_overlapping(time_interval[0],time_interval[1],
                              spindle_start,spindle_end)  
def discritized_onset_label_manual(epochs,raw,epoch_length, df,spindle_duration=2,):
    temporal_event = epochs.events[:,0] / raw.info['sfreq']
    start_times = temporal_event
    end_times = start_times + epoch_length
    discritized_time_intervals = np.vstack((start_times,end_times)).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    temp=[]
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            temp.append([time_interval,spindle])
            #print(time_interval,spindle)
            if spindle_comparison(time_interval,spindle,spindle_duration):
                #print('yes');sleep(4)
                #print(time_interval,spindle-0.5,spindle+1.5)
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,discritized_time_to_zero_one_labels
def discritized_onset_label_auto(epochs,raw,df,epoch_length):
    temporal_event = epochs.events[:,0] / raw.info['sfreq']
    start_times = temporal_event
    end_times = start_times + epoch_length
    discritized_time_intervals = np.vstack((start_times,end_times)).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for kk,(spindle,spindle_duration) in enumerate(zip(df['Onset'],df['Duration'])):
            if spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=False):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,discritized_time_intervals
def spindle_check(x):
    import re
    if re.compile('spindle',re.IGNORECASE).search(x):
        return True
    else:
        return False
