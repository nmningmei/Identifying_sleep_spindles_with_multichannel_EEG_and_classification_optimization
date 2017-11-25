# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:29:50 2017

@author: ning
"""

import numpy as np
import mne
import os
import pandas as pd
from utilities import *
from sklearn import metrics
from tqdm import tqdm

class Filter_based_and_thresholding:
    """
    example of using:
        os.chdir('~\\training set')
        raw_file = 'suj8_d2_nap.fif'
        a_file = 'suj8_d2final_annotations.txt'
        # get the annotations if you have, generating a data frame that contain three columns: Onset, Duration, Annotation
        annotations = pd.read_csv(a_file)
        # use MNE-python to read preprocessed EEG data. A raw object must be read before the later steps
        raw = mne.io.read_raw_fif(raw_file,)
        # initialize the object function
        a=Filter_based_and_thresholding()
        # step one: check the data online
        a.get_raw(raw)
        # step two (optional if you want to output the probability of segmented EEG signals, in which whether it contains a spindle)
        a.get_epochs()
        # step three: read the annotation data frame to the object for later purpose
        a.get_annotation(annotations)
        # step four (main step): use Filter-based and thresholding to find onsets and durations of sleep spindles
        # the only input would be the signal itself and the lower/higher thresholds, no other information added
        a.find_onset_duration()
        # step five (optional): exclude sleep spindles found outside sleep stage 2
        a.sleep_stage_check()
        # step six (optional): output probabilities of segmented signals of whether they contain a spindle
        a.predict_proba()
        # step seven: generate true labels for cross validation or optimization the hyper-parameters
        a.mauanl_label()
    
    Parameters:
        channelList: channels that are interested
        moving_window_size: size of the moving window for convolved root mean square computation. 
        It should work better when it is the sampling frequency, which, in this case is 500 
                (we downsample subjects with 1000 Hz sampling rate).
        lower_threshold: highpass threshold for spindle detection: 
            decision making = trimmed_mean + lower_T * trimmed_std
        higher_threshold: lowpass threshold for spindle detection: 
            decision making = trimmed_mean + higher_T * trimmed_std
        syn_channels: criteria for selecting spindles: at least # of channels have spindle instance and also in the mean channel
        l_bound: low boundary for duration of a spindle instance
        h_bound: high boundary for duration of a spindle instance
        tol: tolerance for determing spindles (criteria in time)
        front: First few seconds of recordings that we are not interested because there might be artifacts, 
        or it is confirmed subjects could not fall asleep within such a short period
        back: last few seconds of recordings that we are not interested due to the recording procedures
        validation_windowsize: the windowsize to segment the data. Unit in seconds. 
        l_freq: high pass frequency
        h_freq: low pass frequency
        
    Attributions:
        __init__: to initialize the parameters
        get_raw: to make sure the EEG data loaded by MNE-python is online for use
        get_annotation: load annotation data frame for later use
        get_epochs: segment the data by Hamming window, and window size is defined in the parameters (validation_windowsize)
        find_onset_duration: use Filter-based and thresholding method to find onests and durations of sleep spindles
        sleep_stage_check: work on the annotation data frame. If sleep stage information is provided in the data frame, 
                we will use it to exclude found spindles that are not in the sleep stage 2
        prepare_validation: work on epoch data. Extra three types of features: 1) signal feature, 
                computed by pseudo- filter-based and thresholding method, 2) frequency feature, the dominant frequency, 
                3) power feature: the power of the dominant frequency
        manual_label: use information provided by the annotation data frame to generate 0,1 labels for cross validating 
                the predicted probabilities/predicted labels, or optimizing the pipeline
    
    """
    def __init__(self,channelList=None,moving_window_size=500,
                syn_channels=3,
                l_bound=0.5,
                h_bound=2,tol=1,
                front=300,back=100,
                validation_windowsize=3,
                l_freq=11,h_freq=16):
        if channelList is None:
            self.channelList = ['F3','F4','C3','C4','O1','O2']
        else:
            self.channelList = channelList
        self.moving_window_size= moving_window_size
        self.syn_channels = syn_channels
        self.l_bound = l_bound
        self.h_bound = h_bound
        self.front = front
        self.back = back
        self.validation_windowsize = validation_windowsize
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.tol = tol
    
    def get_raw(self,raw):
        raw.load_data()
        l_freq,h_freq=self.l_freq,self.h_freq
        raw.pick_channels(self.channelList)
        raw.info.normalize_proj()
        raw.filter(l_freq,h_freq)
        back = self.back
        self.raw = raw
        sfreq = raw.info['sfreq']
        self.moving_window_size = sfreq
        self.sfreq = sfreq
        self.last = raw.last_samp/sfreq - back
        
    def get_annotation(self,annotation):
        annotation = annotation
        self.annotation = annotation
    
    def get_epochs(self):
        from mne.time_frequency import psd_multitaper
        raw = self.raw
        validation_windowsize = self.validation_windowsize
        front = self.front
        back = self.back
        l_freq = self.l_freq
        h_freq = self.h_freq
        events = mne.make_fixed_length_events(raw,id=1,start=front,
                                             stop=raw.times[-1]-back,
                                             duration=validation_windowsize)
        epochs = mne.Epochs(raw,events,event_id=1,tmin=0,tmax=validation_windowsize,
                           preload=True)
        epochs.resample(64)
        psds,freq = psd_multitaper(epochs,fmin=l_freq,
                                        fmax=h_freq,
                                        tmin=0,tmax=validation_windowsize,
                                        low_bias=True,)
        psds = 10 * np.log10(psds)
        self.epochs = epochs
        self.psds = psds
        self.freq = freq
    def find_onset_duration(self,lower_threshold,higher_threshold):
        from scipy.stats import trim_mean,hmean
        self.lower_threshold = lower_threshold
        self.higher_threshold = higher_threshold
        front = self.front
        back = self.back
        raw = self.raw
        channelList = self.channelList
        moving_window_size = self.moving_window_size
        l_bound = self.l_bound
        h_bound = self.h_bound
        tol = self.tol
        syn_channels = self.syn_channels
        

        sfreq=raw.info['sfreq']
        time=np.linspace(0,raw.last_samp/sfreq,raw.last_samp)
        RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
        peak_time={} 
        mph,mpl = {},{}
        
        for ii,names in tqdm(enumerate(channelList)):
            peak_time[names]=[]
            segment,_ = raw[ii,:]
            RMS[ii,:] = window_rms(segment[0,:],moving_window_size) 
            mph[names] = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) 
            mpl[names] = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05)
            pass_ = RMS[ii,:] > mph[names]#should be greater than then mean not the threshold to compute duration
            #pass_ = (RMS[ii,:] > mph[names]) & (RMS[ii,:] < mpl[names])
            up = np.where(np.diff(pass_.astype(int))>0)
            down = np.where(np.diff(pass_.astype(int))<0)
            up = up[0]
            down = down[0]
            if down[0] < up[0]:
                down = down[1:]
            if (up.shape > down.shape) or (up.shape < down.shape):
                size = np.min([up.shape,down.shape])
                up = up[:size]
                down = down[:size]
            C = np.vstack((up,down))
            for pairs in C.T:
                if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                    SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                    if np.max(SegmentForPeakSearching) < mpl[names]:
                        temp_temp_time = time[pairs[0]:pairs[1]]
                        ints_temp = np.argmax(SegmentForPeakSearching)
                        peak_time[names].append(temp_temp_time[ints_temp])
        peak_time['mean'],peak_at,duration=[],[],[]
        RMS_mean = hmean(RMS)
        mph['mean'] = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS_mean,0.05)
        mpl['mean'] = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS_mean,0.05)
        pass_ = RMS_mean > mph['mean']
        #pass_ = (RMS_mean > mph['mean']) & (RMS_mean < mpl['mean'])
        up = np.where(np.diff(pass_.astype(int))>0)
        down= np.where(np.diff(pass_.astype(int))<0)
        up = up[0]
        down = down[0]
        if down[0] < up[0]:
            down = down[1:]
        if (up.shape > down.shape) or (up.shape < down.shape):
            size = np.min([up.shape,down.shape])
            up = up[:size]
            down = down[:size]
        C = np.vstack((up,down))
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
                if np.max(SegmentForPeakSearching)< mpl['mean']:
                    temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time['mean'].append(temp_time[ints_temp])
                    peak_at.append(SegmentForPeakSearching[ints_temp])
                    duration_temp = time[pairs[1]] - time[pairs[0]]
                    duration.append(duration_temp)
        time_find=[];mean_peak_power=[];Duration=[];
        for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
            temp_timePoint=[]
            for ii, names in enumerate(channelList):
                try:
                    temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
                except:
                    temp_timePoint.append(item + 2)
            try:
                if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
                    time_find.append(float(item))
                    mean_peak_power.append(PEAK)
                    Duration.append(duration_time)
                    #print(float(item),PEAK,duration_time)
            except:
                pass
        self.time_find = time_find
        self.mean_peak_power = mean_peak_power
        self.Duration = Duration
        
    def sleep_stage_check(self):
        annotations = self.annotation
        tol = self.tol
        time_find = self.time_find
        mean_peak_power = self.mean_peak_power
        Duration = self.Duration
        front = self.front
        last = self.last
        try:
            temp_time_find=[];temp_mean_peak_power=[];temp_duration=[];
            # seperate out stage 2
            stages = annotations[annotations.Annotation.apply(stage_check)]
            On = stages[::2];Off = stages[1::2]
            stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
            if abs(np.diff(stage_on_off[0]) - 30) < 2:
                pass
            else:
                On = stages[1::2];Off = stages[::2]
                stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
            for single_time_find, single_mean_peak_power, single_duration in zip(time_find,mean_peak_power,Duration):
                for on_time,off_time in stage_on_off:
                    if intervalCheck([on_time,off_time],single_time_find,tol=tol):
                        temp_time_find.append(single_time_find)
                        temp_mean_peak_power.append(single_mean_peak_power)
                        temp_duration.append(single_duration)
            time_find=temp_time_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
            self.time_find = temp_time_find
            self.mean_peak_power = temp_mean_peak_power
            self.Duration = temp_duration
            
        except:
            print('stage 2 missing')
        result = pd.DataFrame({'Onset':time_find,'Duration':Duration,'Annotation':['spindle']*len(Duration)})
        result = result[(result['Onset'] > front) & (result['Onset'] < last)]
        self.auto_scores = result
    def prepare_validation(self,):
        import pandas as pd
        
        channelList = self.channelList
        lower_threshold = self.lower_threshold
        higher_threshold = self.higher_threshold
        time_find,Duration = self.time_find,self.Duration
        epochs = self.epochs
        psds = self.psds
        freq = self.freq
        validation_windowsize = self.validation_windowsize
        raw = self.raw
        
        result = pd.DataFrame({'Onset':time_find,'Duration':Duration,'Annotation':['spindle']*len(Duration)})
        
        data = epochs.get_data()
        full_prop = [[psuedo_rms(lower_threshold,higher_threshold,d[ii,:]) for ii,name in enumerate(channelList)] for d in data]
        
        features = pd.DataFrame(np.concatenate((np.array(full_prop),psds.max(2),freq[np.argmax(psds,2)]),1))
        
        
        auto_label,_ = discritized_onset_label_auto(epochs,raw,result,
                                                 validation_windowsize)
        self.auto_labels = auto_label
        self.decision_features = features
        
    def fit(self,proba_exclude=False,proba_threshold=0.5,n_jobs=1,cv=None,clf=None):
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import cross_val_predict,KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        decision_features = self.decision_features
        auto_labels = self.auto_labels
        if cv is None:
            cv = KFold(n_splits=5,shuffle=True,random_state=12345)
        if clf is None:
            clf = LogisticRegressionCV(Cs=np.logspace(-4,6,11),
                                   cv=cv,
                                   tol=1e-5,
                                   max_iter=int(1e4),
                                   scoring='roc_auc',
                                   class_weight='balanced',
                                   n_jobs=n_jobs)
            clf = Pipeline([('scaler',StandardScaler()),
                        ('estimator',clf)])
        
        try:
            auto_proba = cross_val_predict(clf,decision_features,auto_labels,cv=cv,method='predict_proba',n_jobs=n_jobs)
            auto_proba = auto_proba[:,-1]
        except:
            try:
                auto_proba = cross_val_predict(clf,decision_features,auto_labels,cv=5,method='predict_proba',n_jobs=n_jobs)
                auto_proba = auto_proba[:,-1]
            except:
                
                auto_proba = cross_val_predict(clf,decision_features,auto_labels,cv=3,method='predict_proba',n_jobs=n_jobs)
                auto_proba = auto_proba[:,-1]
        if proba_exclude:
            idx_ = np.where(auto_proba < proba_threshold)
            auto_labels[idx_] = 0
            #auto_proba[idx_]
        self.auto_labels = auto_labels
        self.auto_proba = auto_proba
        
    def make_manuanl_label(self):
        raw = self.raw
        epochs = self.epochs
        annotations = self.annotation
        validation_windowsize = self.validation_windowsize
        
        spindles = annotations[annotations['Annotation'].apply(spindle_check)]
        #print('number of spindles marked: %d' %(len(spindles)))
        manual_labels,_ = discritized_onset_label_manual(epochs,raw,
                                                         validation_windowsize,
                                                         spindles,)
        self.manual_labels = manual_labels
        self.spindles = spindles
"""
if __name__ == "__main__":
    
    os.chdir('D:\\NING - spindle\\training set')
    raw_file = 'suj8_d2_nap.fif'
    a_file = 'suj8_d2final_annotations.txt'
    annotations = pd.read_csv(a_file)        
    raw = mne.io.read_raw_fif(raw_file,)
    a=Filter_based_and_thresholding()
    a.get_raw(raw)
    a.get_epochs()
    a.get_annotation(annotations)
    def cost(params):
        lower_threshold,higher_threshold = params
        a.find_onset_duration(lower_threshold,higher_threshold)
        a.sleep_stage_check()
        a.prepare_validation()
        a.make_manuanl_label()
        a.fit()
        return metrics.roc_auc_score(a.manual_labels,a.auto_proba)
    lower = np.arange(0.1,1.1,0.1)
    higher = np.arange(2.5,3.6,0.1)
    pairs = [(l,h) for l in lower for h in higher]
    result = {'lower_threshold':[],'higher_threshold':[],'roc_auc':[]}
    for p in pairs:
        result['lower_threshold'].append(p[0])
        result['higher_threshold'].append(p[1])
        result['roc_auc'].append(cost(p))
"""    