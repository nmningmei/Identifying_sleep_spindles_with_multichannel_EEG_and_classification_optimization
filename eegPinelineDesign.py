# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:39:38 2016
@author: ning
"""

import numpy as np
import random
import mne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=16); matplotlib.rc('axes', titlesize=16) 
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import re
import scipy
from scipy import signal
import math
from mne.time_frequency import psd_multitaper
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#from obspy.signal.filter import bandpass

def change_file_directory(path_directory):
    '''Change working directory, and return files that exist in the changed to directory'''
    current_directory=os.chdir(path_directory)
    #print(os.listdir(current_directory))
    return os.listdir(current_directory)

def split_type_of_files():
    EEGFind = re.compile("vhdr", re.IGNORECASE);EEGfiles=[]
    TXTFind = re.compile("txt",re.IGNORECASE);Annotationfiles=[]
    """This function will go through the current directory and
    look at all the files in the directory.
        The reason I have this function is because it create a file
    space for looping the feature extraction"""
    """However, we do not use this function any more"""
    directoryList = os.listdir(os.getcwd())
    for item in directoryList:
        if EEGFind.search(item):
            EEGfiles.append(item)
        elif TXTFind.search(item):
            Annotationfiles.append(item)
    return EEGfiles,Annotationfiles



def pick_sample_file(EEGfile,n=0):
    """this function is used as a way to get names for dictionary variables"""
    """It takes an EEG file string and return the string before the dot"""
    file_to_read=EEGfile[n]
    fileName=file_to_read.split('.')[0]
    return file_to_read,fileName

chan_dict={'Ch56': 'TP8', 'Ch61': 'F6', 'Ch3': 'F3', 'Ch45': 'P1', 'Ch14': 'P3', 
           'Ch41': 'C1', 'Ch1': 'Fp1', 'Ch46': 'P5', 'Ch7': 'FC1', 'Ch37': 'F5', 
           'Ch21': 'TP10', 'Ch8': 'C3', 'Ch11': 'CP5', 'Ch28': 'FC6', 'Ch17': 'Oz', 
           'Ch39': 'FC3', 'Ch38': 'FT7', 'Ch58': 'C2', 'Ch33': 'AF7', 'Ch48': 'PO3', 
           'Ch9': 'T7', 'Ch49': 'POz', 'Ch2': 'Fz', 'Ch15': 'P7', 'Ch20': 'P8', 
           'Ch60': 'FT8', 'Ch57': 'C6', 'Ch32': 'Fp2', 'Ch29': 'FC2', 'Ch59': 'FC4', 
           'Ch35': 'AFz', 'Ch44': 'CP3', 'Ch47': 'PO7', 'Ch30': 'F4', 'Ch62': 'F2', 
           'Ch4': 'F7', 'Ch24': 'Cz', 'Ch31': 'F8', 'Ch64': 'ROc', 'Ch23': 'CP2', 
           'Ch25': 'C4', 'Ch40': 'FCz', 'Ch53': 'P2', 'Ch19': 'P4', 'Ch27': 'FT10', 
           'Ch50': 'PO4', 'Ch18': 'O2', 'Ch55': 'CP4', 'Ch6': 'FC5', 'Ch12': 'CP1', 
           'Ch16': 'O1', 'Ch52': 'P6', 'Ch5': 'FT9', 'Ch42': 'C5', 'Ch36': 'F1', 
           'Ch26': 'T8', 'Ch51': 'PO8', 'Ch34': 'AF3', 'Ch22': 'CP6', 'Ch54': 'CPz', 
           'Ch13': 'Pz', 'Ch63': 'LOc', 'Ch43': 'TP7'}
def load_data(file_to_read,low_frequency=.1,high_frequency=50,eegReject=80,
              eogReject=180,n_ch=-2):
    """ not just load the data, but also remove artifact by using mne.ICA
        Make sure 'LOC' or 'ROC' channels are in the channel list, because they
        are used to detect muscle and eye blink movements on EEG data without
        any events, such as sleep EEG recordings
        """
        
        
    """
    file_to_read: file name with 'vhdr'
    low_frequency: high pass cutoff point
    high_frequency: low pass cutoff point
    eegReject: change in amplitude in microvoltage of eeg channels
    eogReject: change in amplitude in microvoltage of eog channels
    n_ch: index of channel list, use '-2' for excluding AUX and stimuli channels
    """
    c=200 # low pass cutoff point used before we proceed to the final data
    # read raw data, scale set to 1 after an update of MNE python
    raw = mne.io.read_raw_brainvision(file_to_read,scale=1,preload=True,)
    raw.set_channel_types({'Aux1':'stim','STI 014':'stim'})
    if 'LOc' in raw.ch_names:# if eye blink channels are in the channel list
        try:
            raw.set_channel_types({'LOc':'eog','ROc':'eog'})
            chan_list=raw.ch_names[:n_ch]#exclude AUX and stim channel
            if 'LOc' not in chan_list:
                chan_list.append('LOc')
            if 'ROc' not in chan_list:
                chan_list.append('ROc')
        
            raw.pick_channels(chan_list)
            # pick only eeg channels to do the first level filtering: low pass filter
            #
            picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
            raw.filter(None,c,l_trans_bandwidth=0.01,
                       h_trans_bandwidth='auto',filter_length=30,picks=picks)
            # compute noise covariance based on continuous data (average references)
            noise_cov=mne.compute_raw_covariance(raw.set_eeg_reference(),picks=picks)# re-referencing to average
            raw.notch_filter(np.arange(60,241,60), picks=picks)
            reject = dict(eeg=eegReject,eog=eogReject)
            # set up ICA on EEG channel data
            ica = mne.preprocessing.ICA(n_components=0.95,n_pca_components =.95,
                                        max_iter=3000,method='extended-infomax',
                                        noise_cov=noise_cov, random_state=0)
            projs, raw.info['projs'] = raw.info['projs'], []
            ica.fit(raw,picks=picks,start=0,stop=raw.last_samp,decim=2,reject=reject,tstep=2.)
            raw.info['projs'] = projs
            ica.detect_artifacts(raw,eog_ch=['LOc','ROc'],
                                 eog_criterion=0.4,skew_criterion=2,kurt_criterion=2,var_criterion=2)
            try:                     
                a,b=ica.find_bads_eog(raw)
                ica.exclude += a
            except:
                pass
        except:
            # same as above, except we filer between 1 and 200 Hz
            print('alternative')
            pass
            raw = mne.io.read_raw_brainvision(file_to_read,scale=1,preload=True)
            raw.resample(500)
            #chan_list=['F3','F4','C3','C4','O1','O2','ROc','LOc']            
            chan_list=raw.ch_names[:n_ch]
            if 'LOc' not in chan_list:
                chan_list.append('LOc')
            if 'ROc' not in chan_list:
                chan_list.append('ROc')
    
            raw.pick_channels(chan_list)
    
            raw.set_channel_types({'LOc':'eog','ROc':'eog'})
            picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=True,stim=False)
            noise_cov=mne.compute_raw_covariance(raw.add_eeg_average_proj(),picks=picks)
            raw.notch_filter(np.arange(60,241,60), picks=picks)
            reject = dict(eeg=eegReject,
                      eog=eogReject)
            raw.filter(1,c)
            raw_proj = mne.compute_proj_raw(raw,n_eeg=1,reject=reject)
            eog_proj,ev = mne.preprocessing.compute_proj_eog(raw,n_eeg=1,average=True,reject=reject,
                                                 l_freq=1,h_freq=c,
                                                 eog_l_freq=1,eog_h_freq=c)
    
            try:
                raw.info['projs'] += eog_proj
            except:
                pass
            raw.info['projs'] += raw_proj
            raw.apply_proj()
            ica = mne.preprocessing.ICA(n_components=0.95,n_pca_components =.95,
                                        max_iter=3000,method='extended-infomax',
                                        noise_cov=noise_cov, random_state=0)
            ica.fit(raw,start=0,stop=raw.last_samp,decim=3,reject=reject,tstep=2.)
            ica.detect_artifacts(raw,eog_ch=['LOc', 'ROc'],eog_criterion=0.4,
                                 skew_criterion=1,kurt_criterion=1,var_criterion=1)
            
            a,b=ica.find_bads_eog(raw)
            ica.exclude += a
    else:
        # if the channel name is not standard, map the list we had above this function
        
        print('no channel names')
        # for some of my data, the scale is strange, but this should be be a
        # main concern for general EEG data
        raw = mne.io.read_raw_brainvision(file_to_read,scale=1e4,preload=True)
        #raw.resample(500)
        raw.rename_channels(chan_dict)
        chan_list=raw.ch_names[:n_ch]
        if 'LOc' not in chan_list:
            chan_list.append('LOc')
        if 'ROc' not in chan_list:
            chan_list.append('ROc')

        raw.pick_channels(chan_list)
        picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
        raw.filter(None,c,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
        noise_cov=mne.compute_raw_covariance(raw.set_eeg_reference(),picks=picks)# re-referencing to average
        raw.notch_filter(np.arange(60,241,60), picks=picks)
        reject = dict(eeg=eegReject,eog=eogReject)
    
        ica = mne.preprocessing.ICA(n_components=0.9,n_pca_components =.9,
                                    max_iter=30,method='extended-infomax',
                                    noise_cov=noise_cov, random_state=0)
        ica.fit(raw,picks=picks,start=0,decim=2,reject=reject,tstep=2.)
        ica.detect_artifacts(raw,eog_ch=['LOc','ROc'],
                             eog_criterion=0.4,skew_criterion=2,kurt_criterion=2,var_criterion=2)
        try:                     
            a,b=ica.find_bads_eog(raw)
            ica.exclude += a
        except:
            pass


    clean_raw = ica.apply(raw,exclude=ica.exclude)
    if low_frequency is not None and high_frequency is not None:
        clean_raw.filter(low_frequency,high_frequency,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
    elif low_frequency is not None or high_frequency is not None:
        try:
            clean_raw.filter(low_frequency,200,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
        except:
            clean_raw.filter(1,high_frequency,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
    else:
        clean_raw = clean_raw
    #clean_raw.plot_psd(fmax=50)
    return clean_raw

def annotation_to_labels(TXTfiles,fileName,label='markon',last_letter=-1):
    """This only works on very particular data structure file."""
    ##################"""This function is no longer used"""###################
    annotation_to_read=[x for x in TXTfiles if fileName in x]
    file = pd.read_csv(annotation_to_read[0])
    labelFind = re.compile(label,re.IGNORECASE)
    windowLabel=[]
    for row in file.iterrows():
        currentEvent = row[1][-1]
        if (labelFind.search(currentEvent)):

            windowLabel.append(currentEvent[-1])
    for idx,items in enumerate(windowLabel):
        if items == ' ':
            windowLabel[idx] = windowLabel[idx -1]
    return windowLabel
def relabel_to_binary(windowLabel,label=['2','3']):
    """This function relabel stage 2 and 3 sleep windows to '1'
    and it is used for classifying sleep stages
    """
    YLabel=[]
    for row in windowLabel:
        if row[0] == label[0] or row[0] == label[1]:
            YLabel.append(1)
        else:
            YLabel.append(0)
    return YLabel
unit_step=lambda x:0 if x<0 else 1
def structure_to_data(channelList,YLabel,raw,sample_points=1000):
    """Become useless after several changes"""
    data={}
    for channel_names in channelList:
        data[channel_names]=[]
    data['label']=[]
    channel_index = mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
    for sample,labels in zip(range(len(YLabel)),YLabel):

        try:
            startPoint=30*sample;endPoint=30*(sample+1)
            start,stop=raw.time_as_index([startPoint,endPoint])
            segment,time=raw[channel_index,start:stop]

            for idx, channel_names in enumerate(channelList):
                yf = 20*np.log10(np.abs(np.fft.rfft(segment[idx,:sample_points])))
                data[channel_names].append(yf)
            data['label'].append(labels)
        except:
            print('last window is missing due to error','sample that is passed is',sample)
            #data['label']=scipy.delete(YLabel,sample,0)
            pass

    return data




def merge_dicts(dict1,dict2):
    """merge two dictionaries if they have the same keys"""
    for key, value in dict2.items():
        dict1.setdefault(key,[]).extend(value)
    return dict1
###################################################################
########### some code for make my own logistic regression #############
def logistic_func(theta, x):
    return 1./(1+np.exp(x.dot(theta)))
def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc
def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)
def grad_desc(theta_values, X, y, lr=10e-8, converge_change=10e-6):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1;#print(i)
    return theta_values, np.array(cost_iter)
def pred_values(theta, X, hard=True,one_sample=False):
    #normalize
    if not one_sample:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob
################## end of logistic regression ###############
########################################################################
def SK_to_data(channelList,markPairs,dataLabels,raw):
    """this function is no longer used because many changes,
    and I lost track of the changes
    """
    data={}
    for channel_names in channelList:
        data[channel_names]=[]
    data['label']=[]
    channel_index,_=dictionary_for_target_channels(channelList,raw)
    for sample,pairs in enumerate(markPairs):
        #print(idx)

        start,stop = raw.time_as_index(pairs)

        segment,time=raw[channel_index,start:stop]
        try:
            for idx,channel_names in enumerate(channelList):
                yf = fft(segment[idx,:]);N=100;#print(channel_names,N)
                data[channel_names].append(np.abs(yf[0:100]))
            data['label'].append(dataLabels[sample])
        except:
            continue

    return data
def annotation_file(TXTFiles,sample_number=0):
    annotation_to_read=[x for x in TXTfiles if fileName in x]
    file = pd.read_csv(annotation_to_read[0])
    file['Duration'] = file['Duration'].fillna(0)
    return file

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """this function is a direct copy from scit kit learn confusion matrix
     tutorial"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()


    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def center_window_by_max_amplitude(raw,time,channelList,windowsWidth=2.0):
    '''The function goes through all channels and return data.frame of
       centered data'''
    """I no longer use this function
    Although this function return the peak of the signal at a window, but 
    because the window was not define as guassian or suitable form, the output
    is questionable
    """
    startPoint=time-windowsWidth;endPoint=time+windowsWidth
    start,stop=raw.time_as_index([startPoint,endPoint])
    tempsegment,timespan=raw[:,start:stop]
    centerxval = timespan[np.argmax(abs(tempsegment[ii,:]))]
    startPoint=centerxval-windowsWidth/2;endPoint=centerxval+windowsWidth/2
    start,stop=raw.time_as_index([startPoint,endPoint])
    segment,_=raw[:,start:stop]
    segment_dictionary={}
    for idx,name in enumerate(channelList):
        yf = fft(segment[idx,:])[:50]
        segment_dictionary[name]= abs(yf)
    return segment_dictionary


def CenterAtPeakOfWindow(timePoint,windowSize,raw,channelIndex):
    '''Simplification of the function above, return only the centered data time
       point.'''
    """simplification of a questionable function, making this one questionable"""
    filter_tempSegment,timeSpan = cut_segments(raw,timePoint,channelIndex)
    peakInd = np.array(find_peaks_cwt(filter_tempSegment[0,:],np.arange(1,500)))
    max_in_peakInd=np.argmax(abs(filter_tempSegment[0,peakInd]))
    centerxval=timeSpan[peakInd[max_in_peakInd]]
    return centerxval

def from_time_markers_to_sample(channelList,raw,windowsWidth=2.0):
    """this function performs simple segmentation of the data,
    The window is a sharp cut window"""
    data={}
    for names in channelList:
        data[names]=[]
    for moments in time:
        segments=center_window_by_max_amplitude(raw,moments, channelList,windowsWidth=windowsWidth)
        for names in channelList:
            data[names.append(segments[names])]
    return data

def standardized(x):
    '''explicit mean centering standardization,
    works only on 1-D vector'''
    normalized_x = (x-np.mean(x))/np.std(x)
    return normalized_x



def add_channels(inst, data, ch_names, ch_types):
    """An unsucessful try on adding extract channel to the existed EEG data"""
    from mne.io import _BaseRaw, RawArray
    from mne.epochs import _BaseEpochs, EpochsArray
    from mne import create_info
    if 'meg' in ch_types or 'eeg' in ch_types:
        return NotImplementedError('Can only add misc, stim and ieeg channels')
    info = create_info(ch_names=ch_names, sfreq=inst.info['sfreq'],
                       ch_types=ch_types)
    if isinstance(inst, _BaseRaw):
        for key in ('buffer_size_sec', 'filename'):
            info[key] = inst.info[key]
        new_inst = RawArray(data, info=info)#, first_samp=inst._first_samps[0])
    elif isinstance(inst, _BaseEpochs):
        new_inst = EpochsArray(data, info=info)
    else:
        raise ValueError('unknown inst type')
    return inst.add_channels([new_inst], copy=True)

def cut_segments(raw,center,channelIndex,windowsize = 1.5):
    """This function takes the center of the signaling window and cut
    a sgement of the signal
    Implementing a Hamming window"""
    startPoint=center-windowsize;endPoint=center+windowsize
    start,stop=raw.time_as_index([startPoint,endPoint])
    tempSegment,timeSpan=raw[channelIndex,start:stop]
    return tempSegment,timeSpan


def Threshold_test(timePoint,raw,channelID,windowsize=2.5):
    """Threshold test implementation of an old paper
    Passing alpha, 11-16 Hz, 30-40 Hz bandpass data with some
    thresholds to determine if a segment of data is dominated by eight one of 
    the above frequencies"""
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    start,stop=raw.time_as_index([startPoint,endPoint])
    se,timeSpan=raw[channelID,start:stop]

    filter_alpha=mne.filter.band_pass_filter(se,1000,8,12)
    filter_spindle=mne.filter.band_pass_filter(se,1000,11,16)
    filter_muscle=mne.filter.band_pass_filter(se,1000,30,40)

    RMS_alpha=np.sqrt(sum(filter_alpha[0,:]**2)/len(filter_alpha[0,:]))
    RMS_spindle=np.sqrt(sum(filter_spindle[0,:]**2)/len(filter_spindle[0,:]))
    RMS_muscle=np.sqrt(sum(filter_muscle[0,:]**2)/len(filter_muscle[0,:]))

    if (RMS_alpha/RMS_spindle <1.2) or (RMS_muscle < 5*10e-4):
        return True
    else:
        return False


def getOverlap(a,b):
    """takes two arrays and return if they are overlapped
    This is a numerical computation. [1,2.2222] is overlaped with
    [2.2222,2.334]"""
    return max(0,min(a[1],b[1]) - max(a[0],b[0]))
def intervalCheck(a,b,tol=0):#a is an array and b is a point
    return a[0]-tol <= b <= a[1]+tol
def spindle_overlapping_test(spindles,timePoint,windowsize,tolerance=0.01):
    """a testing function, and it is no longer used"""
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    return all(getOverlap([startPoint,endPoint],[instance-windowsize,instance+windowsize])<=tolerance for instance in spindles)

def used_windows_check(timePoint,used_time_windows,windowsize,tolerance=0.01):
    """a testing function, and it is no longer used"""
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    return all(getOverlap([startPoint,endPoint],[lower,upper])<=tolerance for (lower,upper) in used_time_windows)

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])


    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def window_rms(a, window_size):
    """One of the core functions
    define a gaussian window based on the given window length (in sample points)
    slide this gaussian window to compute the root mean sqaure of the signal,
    returns an evelope measure of the signal"""
    a2 = np.power(a,2)
    window = signal.gaussian(window_size,(window_size/.68)/2)
    return np.sqrt(np.convolve(a2, window, 'same')/len(a2)) * 1e2


def distance_check(list_of_comparison, time):
    """test function, no longer used"""
    list_of_comparison=np.array(list_of_comparison)
    condition = list_of_comparison - time < 1
    return condition


def RMS_pass(pass_,time,RMS):
    '''Test function, it is no longer used'''
    temp = []
    up = np.where(np.diff(pass_.astype(int))>0)
    down = np.where(np.diff(pass_.astype(int))<0)
    if (up[0].shape > down[0].shape) or (up[0].shape < down[0].shape):
        size = np.min([up[0].shape,down[0].shape])
        up = up[0][:size]
        down = down[0][:size]
    C = np.vstack((up,down))

    for pairs in C.T:
        if 0.5 < (time[pairs[1]] - time[pairs[0]]) < 2:
            #TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
            SegmentForPeakSearching = RMS[pairs[0]:pairs[1]]
            temp_temp_time = time[pairs[0]:pairs[1]]
            ints_temp = np.argmax(SegmentForPeakSearching)
            temp.append(temp_temp_time[ints_temp])

    return temp

def RMS_calculation(intervals,dataSegment,mul):
    """unit test function, and it is no longer used"""
    segment = dataSegment[0,:]
    time = np.linspace(intervals[0],intervals[1],len(segment))
    RMS = window_rms(segment,200)
    mph=scipy.stats.trim_mean(RMS,0.05) + mul * RMS.std()
    pass_=RMS > mph
    peak_time=RMS_pass(pass_,time,RMS)
    return peak_time,RMS,time


def find_time(peak_time,number=3):
    """unit test function, 
    for the spindle onsets found in the mean RMS, how many of them were also
    found in the individual channels?
    1. if nothing was found, do nothing on individual channels eventhough 
        we might have something in the individual channels
    2. else we the one that is closest to the currently looked at onset
    3. using a try-except because I want to avoid empty found in 1."""
    time_find=[]
    for item in peak_time['mean']:
        temp_timePoint=[]
        channelList = ['F3','F4','C3','C4','O1','O2']
        for ii,names in enumerate(channelList):
            if len(peak_time[names]) == 0:
                pass
            else:
                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(float(x[1])-float(item)))[1])
        try:
            if np.sum((abs(np.array(temp_timePoint) - item)<1).astype(int))>number:
                time_find.append(item)
        except:
            pass

    return time_find

def validation(val_file,result,tol=1):
    """match the predicted spindle locally
    if the predicted spindle is not 1 second 
    away from the true spindle, we have a match"""
    file2 = pd.read_csv(val_file,sep=',')
    labelFind = re.compile('spindle',re.IGNORECASE)
    spindles=[]# take existed annotations
    for row in file2.iterrows():
        currentEvent = row[1][-1]
        if labelFind.search(currentEvent):
            spindles.append(row[1][0])# time of marker
    spindles = np.array(spindles)

    peak_time = result['Onset'].values
    Time_found = peak_time
    match=[]
    mismatch=[]
    for item in Time_found:
        if any(abs(item - spindles)<tol):
            match.append(item)
        else:
            mismatch.append(item)
    return spindles, match, mismatch
from scipy.stats import hmean,trim_mean
def EEGpipeline_by_epoch(file_to_read,validation_file,lowCut=10,highCut=18,
                         majority=3,mul=0.8):
    """first end to end attempt of the pipeline.
    """
    raw = mne.io.read_raw_fif(file_to_read,preload=True,add_eeg_ref=False)
    
    raw.filter(lowCut,highCut,l_trans_bandwidth=0.1)
    channelList = ['F3','F4','C3','C4','O1','O2']
    raw.pick_channels(channelList)
    time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at=get_Onest_Amplitude_Duration_of_spindles(raw,
                                                                                                                 channelList,
                                                                                                                 file_to_read,
                                                                                                                 moving_window_size=200,
                                                                                                                 threshold=mul,
                                                                                                                 syn_channels=majority,
                                                                                                                 l_freq=lowCut,
                                                                                                                 h_freq=highCut,
                                                                                                                 l_bound=0.5,
                                                                                                                 h_bound=2)
    
    print('finish loading data')
    file2 = pd.read_csv(validation_file,sep=',')
    labelFind = re.compile('Marker: Markon: 2',re.IGNORECASE)
    stage2=[]# take existed annotations
    for row in file2.iterrows():
        currentEvent = row[1][-1]
        if labelFind.search(currentEvent):
            stage2.append([row[1][0],row[1][0]+30])# time of marker
    stage2 = np.array(stage2)
    print('finish loading annotations')
    result = pd.DataFrame({'Onset':np.array(time_find),'Duration':Duration})
    result['Annotation']='spindle'
    spindles, match, mismatch=validation(val_file=validation_file,result=result,tol=1)

    return peak_time, result,spindles, match, mismatch


def TS_analysis(raw,epch,picks,l_freq=8,h_freq=12):
    """returns power spectral density and frequency correspond to the power spectral"""
    psd_,f=psd_multitaper(raw,tmin=epch[0],tmax=epch[1],fmin=l_freq,fmax=h_freq,picks=picks,n_jobs=-1)
    return psd_,f

def make_overlap_windows(raw,epoch_length=10):
    """get a many row by 2 columns matrix
    each row is the start and end point of the window"""
    candidates = np.arange(raw.first_samp/1000, raw.last_samp/1000,epoch_length/2)
    epochs=[]
    for ii,item in enumerate(candidates):
        #print(ii,len(candidates))
        if ii + 2 > len(candidates)-1:
            break
        else:
            epochs.append([item,candidates[ii+2]])
    return np.array(epochs)


def regressionline(intercept,s,x_range):
    """plot regression line using intercept and s, which is the slope
    only works on one IV regression"""
    try:
        x = np.array(x_range)
        y = intercept*np.ones(len(x)) + s*x
    except:
        y = intercept + s*x
    return y
def epoch_activity(raw,picks,epoch_length=10,l=1,h=200):
    """compute power spectral density over series overlapping windows
    was used for classifying sleep stages
    
    returns delta 1: 1 - 2 Hz
            delta 2: 2 - 4 Hz
            theta: 4 - 8 Hz
            alpha: 8 - 12 Hz
            beta: 12 - 20 Hz
            gamma (low): 20 - 40
            slow spindle: 10 - 12
            fast spindle: 12 - 14
            customized range: l - h Hz"""
    # make epochs based on epoch length (10 secs), and overlapped by half of the window
    epochs = make_overlap_windows(raw,epoch_length=epoch_length)
    
     # preallocate
    alpha_C=[];DT_C=[];ASI=[];activity=[];ave_activity=[];slow_spindle=[];fast_spindle=[]
    psd_delta1=[];psd_delta2=[];psd_theta=[];psd_alpha=[];psd_beta=[];psd_gamma=[];target_spindle=[]

    print('calculating power spectral density')
    for ii,epch in enumerate(epochs):
        # monitor the progress (urgly useless)
        #update_progress(ii,len(epochs))
        psds,f = TS_analysis(raw,epch,picks,0,40)
        psds = psds[0]
        psds = 10*np.log10(psds)#rescale
        temp_psd_delta1 = psds[np.where((f<=2))]# delta 1 band
        temp_psd_delta2 = psds[np.where(((f>=2) & (f<=4)))]# delta 2 band
        temp_psd_theta  = psds[np.where(((f>=4) & (f<=8)))]# theta band
        temp_psd_alpha  = psds[np.where(((f>=8) & (f<=12)))]# alpha band
        temp_psd_beta   = psds[np.where(((f>=12) & (f<=20)))]# beta band
        temp_psd_gamma  = psds[np.where((f>=20))] # gamma band
        temp_slow_spindle = psds[np.where((f>=10) & (f<=12))]# slow spindle
        temp_fast_spindle = psds[np.where((f>=12) & (f<=14))]# fast spindle
        temp_target_spindle = psds[np.where((f>=l) & (f<=h))]

        temp_activity = [temp_psd_delta1.mean(),
                         temp_psd_delta2.mean(),
                         temp_psd_theta.mean(),
                         temp_psd_alpha.mean(),
                         temp_psd_beta.mean(),
                         temp_psd_gamma.mean()]

        temp_ASI = temp_psd_alpha.mean() /( temp_psd_delta2.mean() + temp_psd_theta.mean())

        alpha_C.append(temp_psd_alpha.mean())
        DT_C.append(temp_psd_delta2.mean() + temp_psd_theta.mean())
        ASI.append(temp_ASI)
        ave_activity.append(temp_activity)
        activity.append(psds[:np.where(f<=20)[0][-1]])#zoom in to beta
        slow_spindle.append(temp_slow_spindle)
        fast_spindle.append(temp_fast_spindle)
        target_spindle.append(temp_target_spindle)
        psd_delta1.append(temp_psd_delta1);psd_delta2.append(temp_psd_delta2)
        psd_theta.append(temp_psd_theta);psd_alpha.append(temp_psd_alpha)
        psd_beta.append(temp_psd_beta);psd_gamma.append(temp_psd_gamma)
    slow_range=f[np.where((f>=10) & (f<=12))];fast_range=f[np.where((f>=12) & (f<=14))]
    return target_spindle,alpha_C,DT_C,ASI,activity,ave_activity,psd_delta1,psd_delta2,psd_theta,psd_alpha,psd_beta,psd_gamma,slow_spindle,fast_spindle,slow_range,fast_range,epochs

def mean_without_outlier(data):
    """basically it is scipy.stats.trimmed_mean"""
    outlier_threshold = data.mean() + data.std()*3
    temp_data = data[np.logical_and(-outlier_threshold < data, data < outlier_threshold)]
    return temp_data.mean()
def trimmed_std(data,percentile):
    """basically it is scipy.stats.trimmed_data and take the std"""
    temp=data.copy()
    temp.sort()
    percentile = percentile / 2
    low = int(percentile * len(temp))
    high = int((1. - percentile) * len(temp))
    return temp[low:high].std(ddof=0)
def get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,moving_window_size=200,
                                             threshold=.9,syn_channels=3,l_freq=0,h_freq=200,l_bound=0.5,
                                             h_bound=2,tol=1):
    """First function implement the filter based and thresholding model. This function is first published on
    my OSF page
    
    raw: raw EEG data object, loaded by the MNE python
    channelList: channels of interest
    file_to_read: useless argument, will be removed in the later versions
    moving_window_size: window size for the sliding window. This sliding window is used for computing the evelope
     of the signal. Unit in number of sample points. Be careful of the sampling rate
    threshold: lower threshold. Kinda the lower boundary of the cutoff
    syn_channels: channel agreement criterion
    l_freq: low cutoff frequency
    h_freq: high cutoff frequency
    l_bound: shortest duration of a spindle
    h_bound: longer duration of a spindle
    tol: temporal toleration used with the syn_channels
    """
    mul=threshold;nn=3.5# this becomes one of the arguments in the later version
    # preallocate time series array for later use
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    # preallocate empty matrices (channels by full sample length of the data)
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    peak_time={} #preallocate
    fig=plt.figure(figsize=(40,40))
    ax=plt.subplot(311)
    ax1=plt.subplot(312,sharex=ax)
    ax2=plt.subplot(313,sharex=ax)
    for ii, names in enumerate(channelList):

        peak_time[names]=[]#preallocate empty list for storage
        segment,_ = raw[ii,:] # get data of one channel
        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of some samples
        #I trimmed that std here but not at the mean RMS, what the hell????????
        mph = trim_mean(RMS[ii,100000:-30000],0.05) + mul * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
        mpl = trim_mean(RMS[ii,100000:-30000],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
        pass_ = RMS[ii,:] > mph

        up = np.where(np.diff(pass_.astype(int))>0)# find the intersections where the RMS goes up
        down = np.where(np.diff(pass_.astype(int))<0)# intesections where the RMS goes down
        up = up[0]
        down = down[0]
        ###############################
        #print(down[0],up[0])
        # in some cases, the first point is down, which makes no sense if we want the general shape of convex
        if down[0] < up[0]:
            down = down[1:]
        #print(down[0],up[0])
        #############################
        # after taking care of the beginning, we take care of the end. Making sure that these two lists have the
        # same length ----> will be paired up to be time intervals/windows
        if (up.shape > down.shape) or (up.shape < down.shape):
            size = np.min([up.shape,down.shape])
            up = up[:size]
            down = down[:size]
        C = np.vstack((up,down))
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:# only take those match the duration criterion
            
                #search for the peak of the RMS, not the original singal
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl:# if the peak of the RMS is too high, we also take it out,
                # otherwise, we keep them and save the durations, onset of the peak, and the peak
                    temp_temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time[names].append(temp_temp_time[ints_temp])
                    ax.scatter(temp_temp_time[ints_temp],mph+0.1*mph,marker='s',
                               color='blue')
        ax.plot(time,RMS[ii,:],alpha=0.2,label=names)
        ax2.plot(time,segment[0,:],label=names,alpha=0.3)
        ax2.set(xlabel="time",ylabel="$\mu$V",xlim=(time[0],time[-1]),title=file_to_read[:-5]+' band pass %.1f - %.1f Hz' %(l_freq,h_freq))
        ax.set(xlabel="time",ylabel='RMS Amplitude',xlim=(time[0],time[-1]),title='auto detection on each channels')
        ax1.set(xlabel='time',ylabel='Amplitude')
        ax.axhline(mph,color='r',alpha=0.03)
        ax2.legend();ax.legend()
    # do the same thing to the mean RMS
    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    ax1.plot(time,RMS_mean,color='k',alpha=0.3)
    # here is the part I calculate the boundaries using the lower and higher thresholds
    # I haven't use the trimmed standard deviation yet
    # the next version, I will use the trimmed standard deviation witht he trimmed mean because it makes more sense
    # that way
    mph = trim_mean(RMS_mean[100000:-30000],0.05) + mul * RMS_mean.std()
    mpl = trim_mean(RMS_mean[100000:-30000],0.05) + nn * RMS_mean.std()
    pass_ =RMS_mean > mph
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    ###############################
    #print(down[0],up[0])
    if down[0] < up[0]:
        down = down[1:]
    #print(down[0],up[0])
    #############################
    if (up.shape > down.shape) or (up.shape < down.shape):
        size = np.min([up.shape,down.shape])
        up = up[:size]
        down = down[:size]
    C = np.vstack((up,down))
    for pairs in C.T:
        
        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
            #TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
            if np.max(SegmentForPeakSearching)< mpl:
                temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time['mean'].append(temp_time[ints_temp])
                peak_at.append(SegmentForPeakSearching[ints_temp])
                ax1.scatter(temp_time[ints_temp],mph+0.1*mph,marker='s',color='blue')
                duration_temp = time[pairs[1]] - time[pairs[0]]
                duration.append(duration_temp)
    ax1.axhline(mph,color='r',alpha=1.)
    ax1.set_xlim([time[0],time[-1]])


    time_find=[];mean_peak_power=[];Duration=[]
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
        except:
            pass
    return time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at

def recode_annotation(x):
    """recode the annotation strings to numerical values
    
    w: awake
    1: stage 1
    2: stage 2
    3: stage 3
    SWS: stage 3
    """
    if re.compile(': w',re.IGNORECASE).search(x):
        return 0
    elif re.compile(':w',re.IGNORECASE).search(x):
        return 0
    elif re.compile('1',re.IGNORECASE).search(x):
        return 1
    elif re.compile('2',re.IGNORECASE).search(x):
        return 2
    elif re.compile('SWS',re.IGNORECASE).search(x):
        return 3
    elif re.compile('3',re.IGNORECASE).search(x):
        return 3
    else:
        print('error')
        pass


def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist
def pass_function(distance):
    pass_ = distance > distance.mean()
    up = np.where(np.diff(pass_.astype(int))>0)
    down = np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    ###############################
    #print(down[0],up[0])
    if down[0] < up[0]:
        down = down[1:]
    #print(down[0],up[0])
    #############################
    if (up.shape > down.shape) or (up.shape < down.shape):
        size = np.min([up.shape,down.shape])
        up = up[:size]
        down = down[:size]
    C = np.vstack((up,down))
    
    return C
def pass_(distance):
    
    C = pass_function(distance)
    
    up = C[0,:];down=C[1,:]
    tempD= np.inf
    for u,d in zip(up,down):
        if u - np.argmax(distance)<tempD:
            tempD = np.abs(u - np.argmax(distance))
            up_ = u
            down_=d
    return (up_,down_)
from scipy.stats import linregress
def find_tilted_peak(x,y):
    """for power spectral density plots, they are some curves that are tilted
    This function is to fit a regression by tilting the x axis to match the 
    titled degree
    the fitting of the regression is done on a subset of the data because I 
    wanted to look at only the range of the spindle frequency
    """
    x=x[:100];y=y[:100]
    s,intercept,_,_,_ = linregress(x=x,y=y)
    y_pre = regressionline(intercept,s,x)

    A = y - y_pre
    A = A[10:]
    try:
        C = pass_(A)
    except:
        C = (np.argmax(A)-20,np.argmax(A)+20)
    idx_freq_range=C
    idx_devia = np.max([np.abs(np.argmax(A)-idx_freq_range[0]+10),np.abs(np.argmax(A)-idx_freq_range[1]+10)])
    maxArg = np.argmax(A)+10
    #print(maxArg,idx_devia)
    return idx_devia,maxArg
    
#def spindle_validation_step1(raw,channelList,moving_window_size=200,
#                             threshold=.9,syn_channels=3,l_freq=0,h_freq=200,
#                             l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.5,front=300,
#                             back=100):
#    """repetition of FBT function, no longer used"""
#    nn=higher_threshold
#    
#    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
#    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
#    peak_time={} #preallocate
#    sfreq=raw.info['sfreq']
#    for ii, names in enumerate(channelList):
#
#        peak_time[names]=[]
#        segment,_ = raw[ii,:]
#        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of 200ms
#        mph = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + threshold * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
#        mpl = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
#        pass_ = RMS[ii,:] > mph#should be greater than then mean not the threshold to compute duration
#
#        up = np.where(np.diff(pass_.astype(int))>0)
#        down = np.where(np.diff(pass_.astype(int))<0)
#        up = up[0]
#        down = down[0]
#        ###############################
#        #print(down[0],up[0])
#        if down[0] < up[0]:
#            down = down[1:]
#        #print(down[0],up[0])
#        #############################
#        if (up.shape > down.shape) or (up.shape < down.shape):
#            size = np.min([up.shape,down.shape])
#            up = up[:size]
#            down = down[:size]
#        C = np.vstack((up,down))
#        for pairs in C.T:
#            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
#                #TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
#                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
#                if np.max(SegmentForPeakSearching) < mpl:
#                    temp_temp_time = time[pairs[0]:pairs[1]]
#                    ints_temp = np.argmax(SegmentForPeakSearching)
#                    peak_time[names].append(temp_temp_time[ints_temp])
#                    
#        
#
#    peak_time['mean']=[];peak_at=[];duration=[]
#    RMS_mean=hmean(RMS)
#    
#    mph = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + threshold * RMS_mean.std()
#    mpl = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + nn * RMS_mean.std()
#    pass_ =RMS_mean > mph
#    up = np.where(np.diff(pass_.astype(int))>0)
#    down= np.where(np.diff(pass_.astype(int))<0)
#    up = up[0]
#    down = down[0]
#    ###############################
#    #print(down[0],up[0])
#    if down[0] < up[0]:
#        down = down[1:]
#    #print(down[0],up[0])
#    #############################
#    if (up.shape > down.shape) or (up.shape < down.shape):
#        size = np.min([up.shape,down.shape])
#        up = up[:size]
#        down = down[:size]
#    C = np.vstack((up,down))
#    for pairs in C.T:
#        
#        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
#            #TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
#            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
#            if np.max(SegmentForPeakSearching)< mpl:
#                temp_time = time[pairs[0]:pairs[1]]
#                ints_temp = np.argmax(SegmentForPeakSearching)
#                peak_time['mean'].append(temp_time[ints_temp])
#                peak_at.append(SegmentForPeakSearching[ints_temp])
#                duration_temp = time[pairs[1]] - time[pairs[0]]
#                duration.append(duration_temp)
#    
#
#
#    time_find=[];mean_peak_power=[];Duration=[]
#    for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
#        temp_timePoint=[]
#        for ii, names in enumerate(channelList):
#            try:
#                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
#            except:
#                temp_timePoint.append(item + 2)
#        try:
#            if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
#                time_find.append(float(item))
#                mean_peak_power.append(PEAK)
#                Duration.append(duration_time)
#        except:
#            pass
#    
#        
#    return time_find,mean_peak_power,Duration,peak_time,peak_at
#def spindle_validation_with_sleep_stage(raw,channelList,annotations,moving_window_size=200,threshold=.9,
#                                        syn_channels=3,l_freq=0,h_freq=200,l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.5,
#                                        front=300,back=100):
#    """repetition, just add sleep stage argument"""
#    nn=higher_threshold
#    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
#    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
#    peak_time={} #preallocate
#    sfreq=raw.info['sfreq']
#    # seperate out stage 2
#    stages = annotations[annotations.Annotation.apply(stage_check)]
#    On = stages[::2];Off = stages[1::2]
#    stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
#    if abs(np.diff(stage_on_off[0]) - 30) < 2:
#        pass
#    else:
#        On = stages[1::2];Off = stages[::2]
#        stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
#
#    for ii, names in enumerate(channelList):
#
#        peak_time[names]=[]
#        segment,_ = raw[ii,:]
#        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of 200ms
#        mph = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + threshold * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
#        mpl = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
#        pass_ = RMS[ii,:] > mph#should be greater than then mean not the threshold to compute duration
#
#        up = np.where(np.diff(pass_.astype(int))>0)
#        down = np.where(np.diff(pass_.astype(int))<0)
#        up = up[0]
#        down = down[0]
#        ###############################
#        #print(down[0],up[0])
#        if down[0] < up[0]:
#            down = down[1:]
#        #print(down[0],up[0])
#        #############################
#        if (up.shape > down.shape) or (up.shape < down.shape):
#            size = np.min([up.shape,down.shape])
#            up = up[:size]
#            down = down[:size]
#        C = np.vstack((up,down))
#        for pairs in C.T:
#            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
#                #TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
#                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
#                if np.max(SegmentForPeakSearching) < mpl:
#                    temp_temp_time = time[pairs[0]:pairs[1]]
#                    ints_temp = np.argmax(SegmentForPeakSearching)
#                    peak_time[names].append(temp_temp_time[ints_temp])
#                    
#        
#
#    peak_time['mean']=[];peak_at=[];duration=[]
#    RMS_mean=hmean(RMS)
#    
#    mph = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + threshold * RMS_mean.std()
#    mpl = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + nn * RMS_mean.std()
#    pass_ =RMS_mean > mph
#    up = np.where(np.diff(pass_.astype(int))>0)
#    down= np.where(np.diff(pass_.astype(int))<0)
#    up = up[0]
#    down = down[0]
#    ###############################
#    #print(down[0],up[0])
#    if down[0] < up[0]:
#        down = down[1:]
#    #print(down[0],up[0])
#    #############################
#    if (up.shape > down.shape) or (up.shape < down.shape):
#        size = np.min([up.shape,down.shape])
#        up = up[:size]
#        down = down[:size]
#    C = np.vstack((up,down))
#    for pairs in C.T:
#        
#        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
#            #TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
#            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
#            if np.max(SegmentForPeakSearching)< mpl:
#                temp_time = time[pairs[0]:pairs[1]]
#                ints_temp = np.argmax(SegmentForPeakSearching)
#                peak_time['mean'].append(temp_time[ints_temp])
#                peak_at.append(SegmentForPeakSearching[ints_temp])
#                duration_temp = time[pairs[1]] - time[pairs[0]]
#                duration.append(duration_temp)
#    
#
#
#    time_find=[];mean_peak_power=[];Duration=[]
#    for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
#        temp_timePoint=[]
#        for ii, names in enumerate(channelList):
#            try:
#                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
#            except:
#                temp_timePoint.append(item + 2)
#        try:
#            if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
#                time_find.append(float(item))
#                mean_peak_power.append(PEAK)
#                Duration.append(duration_time)
#        except:
#            pass
#    temp_time_find=[];temp_mean_peak_power=[];temp_duration=[];
#    for single_time_find, single_mean_peak_power, single_duration in zip(time_find,mean_peak_power,Duration):
#        for on_time,off_time in stage_on_off:
#            if intervalCheck([on_time,off_time],single_time_find,tol=tol):
#                temp_time_find.append(single_time_find)
#                temp_mean_peak_power.append(single_mean_peak_power)
#                temp_duration.append(single_duration)
#    time_find=temp_time_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
#    return time_find,mean_peak_power,Duration,peak_time,peak_at
def spindle_validation_with_sleep_stage_after_wavelet_transform(raw,channelList,
                                                                file_to_read,annotations,
                                                                moving_window_size=200,
                                                                threshold=.9,
                                                                syn_channels=3,
                                                                l_freq=0,h_freq=200,
                                                                l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.5):
    """implement wavelet tranform as one of the steps of the processing steps"""
    nn=higher_threshold
    
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    widths = np.arange(1,11)
    peak_time={} #preallocate
    # seperate out stage 2
    stages = annotations[annotations.Annotation.apply(stage_check)]
    On = stages[::2];Off = stages[1::2]
    stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
    if abs(np.diff(stage_on_off[0]) - 30) < 2:
        pass
    else:
        On = stages[1::2];Off = stages[::2]
        stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))

    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        segment,_ = raw[ii,:]
        cwtmatr = signal.cwt(segment[0,:],signal.morlet,widths)
        RMS[ii,:] = window_rms(cwtmatr[0,:],moving_window_size)
        #RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of 200ms
        mph = trim_mean(RMS[ii,100000:-30000],0.05) + threshold * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
        mpl = trim_mean(RMS[ii,100000:-30000],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
        pass_ = RMS[ii,:] > mph#should be greater than then mean not the threshold to compute duration

        up = np.where(np.diff(pass_.astype(int))>0)
        down = np.where(np.diff(pass_.astype(int))<0)
        up = up[0]
        down = down[0]
        ###############################
        #print(down[0],up[0])
        if down[0] < up[0]:
            down = down[1:]
        #print(down[0],up[0])
        #############################
        if (up.shape > down.shape) or (up.shape < down.shape):
            size = np.min([up.shape,down.shape])
            up = up[:size]
            down = down[:size]
        C = np.vstack((up,down))
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                #TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl:
                    temp_temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time[names].append(temp_temp_time[ints_temp])
                    
        

    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    
    mph = trim_mean(RMS_mean[100000:-30000],0.05) + threshold * RMS_mean.std()
    mpl = trim_mean(RMS_mean[100000:-30000],0.05) + nn * RMS_mean.std()
    pass_ =RMS_mean > mph
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    ###############################
    #print(down[0],up[0])
    if down[0] < up[0]:
        down = down[1:]
    #print(down[0],up[0])
    #############################
    if (up.shape > down.shape) or (up.shape < down.shape):
        size = np.min([up.shape,down.shape])
        up = up[:size]
        down = down[:size]
    C = np.vstack((up,down))
    for pairs in C.T:
        
        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
            #TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
            if np.max(SegmentForPeakSearching)< mpl:
                temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time['mean'].append(temp_time[ints_temp])
                peak_at.append(SegmentForPeakSearching[ints_temp])
                duration_temp = time[pairs[1]] - time[pairs[0]]
                duration.append(duration_temp)
    


    time_find=[];mean_peak_power=[];Duration=[]
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
        except:
            pass
    temp_time_find=[];temp_mean_peak_power=[];temp_duration=[];
    for single_time_find, single_mean_peak_power, single_duration in zip(time_find,mean_peak_power,Duration):
        for on_time,off_time in stage_on_off:
            if intervalCheck([on_time,off_time],single_time_find,tol=tol):
                temp_time_find.append(single_time_find)
                temp_mean_peak_power.append(single_mean_peak_power)
                temp_duration.append(single_duration)
    time_find=temp_time_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
    return time_find,mean_peak_power,Duration,peak_time,peak_at
def thresholding_filterbased_spindle_searching(raw,channelList,annotations,moving_window_size=200,lower_threshold=.4,
                                        syn_channels=3,l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.5,
                                        front=300,back=100,sleep_stage=True,proba=False,validation_windowsize=3,l_freq=11,h_freq=16):
    
    """One of the core functions
    raw: data after preprocessing
    channelList: channel list of interest, and in this study we use       'F3','F4','C3','C4','O1','O2'
    annotations: pandas DataFrame object containing manual annotations, such as sleep stages, spindle locations.
    moving_window_size: size of the moving window for convolved root mean square computation. It should work better when it is the sampling frequency, which, in this case is 500 (we downsample subjects with 1000 Hz sampling rate). 
    lower_threshold: highpass threshold for spindle detection: decision making = trimmed_mean + lower_T * trimmed_std
    higher_threshold: lowpass threshold for spindle detection: decision making = trimmed_mean + higher_T * trimmed_std
    syn_channels: criteria for selecting spindles: at least # of channels have spindle instance and also in the mean channel
    l_bound: low boundary for duration of a spindle instance
    h_bound: high boundary for duration of a spindle instance
    tol : tolerance for determing spindles (criteria in time)
    front : First few seconds of recordings that we are not interested because there might be artifacts, or it is confirmed subjects could not fall asleep within such a short period
    back : last few seconds of recordings that we are not interested due to the recording procedures
    
    returns:
        time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label
    
    time_find: onset of spindles. Marked at the peak of the spindle
    mean_peak_power: mean of the peaks of the RMS, not the signal peaks
    Duration: duration of spindles. Marked at each spindle found
    mph: lower threshold
    mpl: higher threshold
    auto_proba: probabilities of whether segmented data contain spindle signals. Provided by a fit logistic
        regression classifier
    auto_label: binary labels (predicted labels) of the segmented data. 1 means the segmented data contains
         spindle signals, else it does not.
        
    """
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    peak_time={} #preallocate
    sfreq=raw.info['sfreq']
    mph,mpl = {},{}
    #########################################################################################################
    ########################### compute the RMSs of the individual channels ###################
    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        segment,_ = raw[ii,:]
        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) 
        mph[names] = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) 
        mpl[names] = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05)
        pass_ = RMS[ii,:] > mph[names]#should be greater than then mean not the threshold to compute duration

        up = np.where(np.diff(pass_.astype(int))>0)
        down = np.where(np.diff(pass_.astype(int))<0)
        up = up[0]
        down = down[0]
        ###############################
        #print(down[0],up[0])
        if down[0] < up[0]:
            down = down[1:]
        #print(down[0],up[0])
        #############################
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
    ########################## finish individual channels #############################################
    ####################################################################################################                
        
    ######################################################################################################
    ################## compute mean of RMSs of the individual channels ######################
    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    
    mph['mean'] = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS_mean,0.05)
    mpl['mean'] = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS_mean,0.05)
    pass_ =RMS_mean > mph['mean']
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    ###############################
    #print(down[0],up[0])
    if down[0] < up[0]:
        down = down[1:]
    #print(down[0],up[0])
    #############################
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
            
    ###################### finish mean RMS ################################################
    #######################################################################################
    
    ##########################################################################################
    ###################### for each found spindle in the mean RMS, compare them to individual
    ###################### channels. If 3 or more channels found spindles at the similar 
    ###################### time, we say at the found time in the mean RMS, we have a spindle
    ###################### a spindle will be marked at the peak of the mean RMS
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
        except:
            pass
    if sleep_stage:# exclude those are not in the sleep stage 2
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
        except:
            print('stage 2 missing')
    
    ################################################################################################
    #################### this is optional 
    #################### we make 3 kinds of features: local RMS thresholding, dominant power spectral density
    #################### and dominant frequency
    #################### the features are extracted on segmented signals (some seconds long and Hamming windowed)
    #################### features are mean centerd and unit standard, fit to a logistic regression classifier
    #################### labels are made based only on the onsets and durations we found in the previous step,
    #################### no information from the manually scores is used!!!!!!!!!!!!!!!!!!!!!!!!!!
    #################### the probabilities of whether segmented data contain spindle signals
    decision_features=None;auto_proba=None;auto_label=None
    if proba:
        print('start probability computing')
        result = pd.DataFrame({'Onset':time_find,'Duration':Duration,'Annotation':['spindle']*len(Duration)})   
        print('making labels')
        auto_label,_ = discritized_onset_label_auto(raw,result,validation_windowsize)
        print('segmenting data')
        events = mne.make_fixed_length_events(raw,id=1,start=front,stop=raw.times[-1]-back,duration=validation_windowsize)
        epochs = mne.Epochs(raw,events,event_id=1,tmin=0,tmax=validation_windowsize,preload=True)
        epochs.resample(100,window='hamming',n_jobs=4)
        data = epochs.get_data()[:,:,:-1]
        full_prop=[]    
        print('gethering self-defined features')
        for d in data:    
            temp_p=[]
            #fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
            for ii,(name) in enumerate(zip(channelList)):#,ax.flatten())):
                rms = window_rms(d[ii,:],500)
                l = trim_mean(rms,0.05) + lower_threshold * trimmed_std(rms,0.05)
                h = trim_mean(rms,0.05) + higher_threshold * trimmed_std(rms,0.05)
                prop = (sum(rms>l)+sum(rms<h))/(sum(rms<h) - sum(rms<l))
                if np.isinf(prop):
                    prop = (sum(rms>l)+sum(rms<h))
                temp_p.append(prop)
                
            
            full_prop.append(temp_p)
        print('computing power spectral density')
        psds,freq = mne.time_frequency.psd_multitaper(epochs,fmin=l_freq,fmax=h_freq,tmin=0,tmax=3,low_bias=True,n_jobs=4)
        psds = 10* np.log10(psds)
        features = pd.DataFrame(np.concatenate((np.array(full_prop),psds.max(2),freq[np.argmax(psds,2)]),1))
        print('standardize')
        decision_features = StandardScaler().fit_transform(features.values,auto_label)
        #clf = LogisticRegressionCV(Cs=np.logspace(-4,6,11),cv=5,tol=1e-4,max_iter=int(1e7))
        clf = SGDClassifier(loss='modified_huber',class_weight='balanced',random_state=12345)
        print('fitting a model')
        clf.fit(decision_features,auto_label)
        print('output probability of each segmented data')
        auto_proba=clf.predict_proba(decision_features)[:,-1]
            
    return time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    """One of the core functions
    
    time_interval: [start time, end time]
    spindle: if spindle_duration_fix is true, this argument is one of the manually marked onsets of the spindles
    spindle_duration: defined spindle duration for manually marked spindles:: 2 seconds
    spindle_duration_fix: if true, meaning we are passing manually scored spindles, else, we are passing
                        automatically scored spindles
                        
    Returns True if the spindle represeted by the spindle onset is overlap the time interval
    Returns False if not
    """
    if spindle_duration_fix:
        spindle_start = spindle - 0.5
        spindle_end   = spindle + 1.5
        a =  np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        """ || ==================|| time_interval
                ||=======================|| and spindle start is in the time interval"""
                
        """||===================|| time_interval
        ||===============|| and spindle end is in the time interval"""
        return a
    else:
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        a = np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
def discritized_onset_label_manual(raw,df,spindle_segment,front=300,back=100):
    """One of the core functions
    raw: raw data object, loaded by MNE python
    df: data frame contains manually scored spindle annotations
    spindle_segment: the length of sliding window used for segment the data
    front: first few seconds of signal cut off from the analysis
    back: last few seconds of the signal cut off from the analysis
    
    returns a list of [0 1] labels. 1 means the segment overlaps a manually marked spindle
                                    0 means not
    """
    # 1-D vector of start times of the sliding non-overlapping window
    discritized_continuous_time = np.arange(front,raw.times[-1]-back,step=spindle_segment)
    # the next start time of the sliding window is the end time of the previous one
    # stack them together, and we will get a 2-D matrix where each row is a time interval
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    # transpose
    discritized_time_intervals = np.array(discritized_time_intervals).T
    # preallocation
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    temp=[] # sanity check
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]# redundant line
        for spindle in df['Onset']:
            temp.append([time_interval,spindle])
            if spindle_comparison(time_interval,spindle,spindle_segment):# if the time interval overlaps a spindle (any spindle), with assuming the spindle is 2 seconds long
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,temp
def discritized_onset_label_auto(raw,df,spindle_segment,front=300,back=100):
    """One of the core functions
    
    raw: raw data object
    df: data frame contains auto-marked spindle information: onsets, durations, peaks
    spindle_segment: window size. Unit: seconds
    front: see above
    back: see above
    
    returns a list of [0 1] labels. 1 means the segment overlaps a automatically marked spindle
                                    0 means not
    """
    spindle_duration = df['Duration'].values
    discritized_continuous_time = np.arange(front,raw.times[-1]-back,step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for kk,spindle in enumerate(df['Onset']):
            if spindle_comparison(time_interval,spindle,spindle_duration[kk],spindle_duration_fix=False):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,discritized_time_intervals

def read_annotation(raw, annotation_file,front=300,back=100):
    """One of the core functions
    The function reads annotation txt file and returns a dataframe containing all the onsets of different
    events: sleep stages, k-complexes, spindles...
    Taking the raw object to the function is to know where to put the 'cut-off' time: the first 300 and last 100 seconds
    And then, only the spindle annotations are selected to form the dataframe of the gold standard
    
    """
    try:# sometimes, I put the file name in list, so I shall take it out
        manual_spindle = pd.read_csv(annotation_file[0])
    except:
        manual_spindle = pd.read_csv(annotation_file)
    manual_spindle = manual_spindle[manual_spindle.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
    manual_spindle = manual_spindle[manual_spindle.Onset > front] 
    keyword = re.compile('spindle',re.IGNORECASE)
    gold_standard = {'Onset':[],'Annotation':[]}
    for ii,row in manual_spindle.iterrows():
        if keyword.search(row[-1]):
            gold_standard['Onset'].append(float(row.Onset))
            gold_standard['Annotation'].append(row.Annotation)
    gold_standard = pd.DataFrame(gold_standard) 
    return gold_standard 
def stage_check(x):
    """A simple function to chack if a string contains keyword '2'"""
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
def sample_data(time_interval_1,time_interval_2,raw,raw_data,stage_on_off,key='miss',old=False):
    """This function is used to be one of the core functions, and sample data in terms of its key
    For example, if I want to sample the segments that are missed by the FBT model, the function will
    perform a cross validation to sample them. 
    
    But, it is no longer used
    """
    if old:
        if key == 'miss':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,1
            else:
                return [[],[]]
        elif key == 'hit':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,1
            else:
                return [[],[]]
        elif key == 'fa':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,0
            else:
                return [[],[]]
        elif key == 'cr':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,0
            else:
                return [[],[]]
    
    else:
        if key == 'miss':
            
            if (sum(intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,1
                else:
                    return [[],[]]
            else:
                return [[],[]]
        elif key == 'hit':
            
            if (sum(intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,1
                else:
                    return [[],[]]
            else:
                return [[],[]]
        elif key == 'fa':
            
            if (sum(intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,0
                else:
                    return [[],[]]
            else:
                return [[],[]]
        elif key == 'cr':
            
            if (sum(intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,0
                else:
                    return [[],[]]
            else:
                return [[],[]]
#def sampling_FA_MISS_CR(comparedRsult,manual_labels, raw, annotation, discritized_time_intervals,samples,label,old,
#                        front=300,back=100):
#    idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
#    idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
#    idx_miss= np.where(comparedRsult == 1)[0]
#    idx_FA  = np.where(comparedRsult == -1)[0]
#    raw_data, time = raw[:,front*raw.info['sfreq']:-back*raw.info['sfreq']]
#    stages = annotation[annotation.Annotation.apply(stage_check)]
#    
#    On = stages[::2];Off = stages[1::2]
#    stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
#    if abs(np.diff(stage_on_off[0]) - 30) < 2:
#        pass
#    else:
#        On = stages[1::2];Off = stages[::2]
#        stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
#
#    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_miss]):
#        a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='miss', old=old)
#
#        if len(a) > 0:
#            a = a.tolist()
#            #print(len(a))
#            samples.append(a)
#            label.append(c)
#
#    for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_FA]):
#
#        a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='fa', old=old)
#        if len(a) > 0:
#            a = a.tolist()
#            #print(len(a))
#            samples.append(a)
#            label.append(c)
#    
#    b = abs(len(idx_miss) - len(idx_FA))
#    if b > 0:
#        for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_CR][:b]):
#            a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='cr', old=old)
#            if len(a) > 0:
#                a = a.tolist()
#                #print(len(a))
#                samples.append(a)
#                label.append(c)
#    elif b < 0:
#        for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_hit][:b]):
#            a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='hit', old=old)
#            if len(a) > 0:
#                a = a.tolist()
#                #print(len(a))
#                samples.append(a)
#                label.append(c)
#    
#    return samples,label
#def mark_stage_2_FA(discritized_time_intervals,stage_on_off,idx_FA):
#    """no longer used due to its lack of readable logic"""
#    t_idx=[]
#    for t1,t2 in discritized_time_intervals[idx_FA]:
#        a=[(t1 > np.array(stage_on_off)[:,0]).astype(int),(t2<np.array(stage_on_off)[:,1]).astype(int)]
#        a = np.array(a)
#        try:
#            t_idx.append(np.where(a.sum(0)==2)[0][0])
#        except:
#            pass
#    return np.unique(t_idx)
def sampling_FA_MISS_CR(comparedResult,manual_labels, raw, annotation, discritized_time_intervals,sample,label,front=300,back=100,):
    """
    One of the core functions
    
    Inputs:
        comparedResult: a 1-D array computed by subtracting auto label vector from the manual label vector, element wise
        manual_labels: a 1-D array, labeling segmented epochs based on manually scored annotations. 
        raw: raw EEG object loaded using MNE python
        annotation: dataframe contains the manually scored annotations
        discritized_time_intervals: a 2-D array, each row represents time window we are going to look at in the loop
        sample: a list. It is used to store processed features
        label: a list. it is used to store labels. The labels are made only based on the true labels. Thus, false alarm 
            and correct rejection are non-spindles (0), while hit and miss are spindles (1)
        front, back: time in seconds we will cut from the original signal
        
        
    Return:
        sample: list of processed features
        label: 0 or 1, indicating the classes of the sampled features
    
    """
    """ for example: auto label = [1,0,1,0] and manual label = [1,1,0,0]
                    comparedResult = [0,1,-1,0] --> hit, miss, false alarm, correc reject
    """
    idx_hit = np.where(np.logical_and((comparedResult == 0),(manual_labels == 1)))[0]
    idx_CR  = np.where(np.logical_and((comparedResult == 0),(manual_labels == 0)))[0]
    idx_miss= np.where(comparedResult == 1)[0]
    idx_FA  = np.where(comparedResult == -1)[0]
    stages = annotation[annotation.Annotation.apply(stage_check)]
    # time stamps of on and off of each stage 2 sleep
    # the if-else was used because some of annotations were strange. Some of them have the onset of stage 2, but no off set
    On = stages[::2];Off = stages[1::2]
    stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
    if abs(np.diff(stage_on_off[0]) - 30) < 2:
        pass
    else:
        On = stages[1::2];Off = stages[::2]
        stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
    # importance: the stop time point is the end of signal subtracts the back cut off
    stop = raw.times[-1]-back
    events = mne.make_fixed_length_events(raw,1,start=front,stop=stop,duration=3,)
    # apply events to raw signal and segment the continuous data to 3-second long epochs
    epochs = mne.Epochs(raw,events,1,tmin=0,tmax=3,proj=False,preload=True)
    # compute the power spectral density and frequency 
    psds, freqs=mne.time_frequency.psd_multitaper(epochs,tmin=0,tmax=3,low_bias=True,proj=False,)
    psds = 10* np.log10(psds)
    data = epochs.get_data()[:,:,:-1];freqs = freqs[psds.argmax(2)];psds = psds.max(2); 
    freqs = freqs.reshape(len(freqs),6,1);psds = psds.reshape(len(psds),6,1)
    # concatanate the signal features
    data = np.concatenate([data,psds,freqs],axis=2)
    # for each segment, we vectorize the features
    data = data.reshape(len(events),-1)
    # see how I only take the miss and FA cases?
    sample.append(data[idx_miss]);label.append(np.ones(len(idx_miss)))
    sample.append(data[idx_FA]);label.append(np.zeros(len(idx_FA)))
    # to make the spindle and non-spindle cases more balanced, we add some instances of the hit and/or correct rejection
    len_need = len(idx_FA) - len(idx_miss)
    if len_need > 0:# if we have more non spindles than spindles
        try:
            idx_hit_need = np.random.choice(idx_hit,size=len_need,replace=False)
        except:
            idx_hit_need = np.random.choice(idx_hit,size=len_need,replace=True)
        sample.append(data[idx_hit_need])
        label.append(np.ones(len(idx_hit_need)))
    else: # else if we have more spindles than non spindles
        idx_CR_nedd = np.random.choice(idx_CR,len_need,replace=False)
        sample.append(data[idx_CR_nedd])
        label.append(np.zeros(len(idx_CR_nedd)))
    return sample,label# so the way I collect the data was to ignore subject or day

def data_gathering_pipeline(temp_dictionary,
                            sampling,
                            labeling,do='with_stage',sub='11',day='day1',
                             raw=None,channelList=None,
                            file=None,windowSize=500,
                            threshold=0.6,syn_channel=3,
                            l=1,h=40,annotation=None,old=True,annotation_file=None,higher_threshold=1.,
                            front=300,back=100):
    """
    A wrapper for the data gathering pipeline.
    Make use of other functions so I just get the results
    """
    if do == 'with_stage':
        time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=thresholding_filterbased_spindle_searching(raw,channelList,annotation,
                                                                                                                    moving_window_size=windowSize,
                                                                                                                    lower_threshold=threshold,higher_threshold=higher_threshold,
                                                                                                                    proba=False,front=front,back=back,
                                                                                                                    sleep_stage=True)
#        time_find,mean_peak_power,Duration,peak_time,peak_at=spindle_validation_with_sleep_stage(raw,
#                                                                                                 channelList,annotation,
#                                                                                                 moving_window_size=windowSize,
#                                                                                                 threshold=threshold,
#                                                                                                 syn_channels=syn_channel,
#                                                                                                 l_freq=l,
#                                                                                                 h_freq=h,
#                                                                                                 l_bound=0.5,
#                                                                                                 h_bound=3.0,tol=1,
#                                                                                                 higher_threshold=higher_threshold,
#                                                                                                 )
        
#    elif do == 'without_stage':
#        time_find,mean_peak_power,Duration,peak_time,peak_at=spindle_validation_step1(raw,
#                                                                                     channelList,
#                                                                                     moving_window_size=windowSize,
#                                                                                     threshold=threshold,
#                                                                                     syn_channels=syn_channel,
#                                                                                     l_freq=l,
#                                                                                     h_freq=h,
#                                                                                     l_bound=0.5,
#                                                                                    h_bound=3.0,tol=1,
#                                                                                    higher_threshold=higher_threshold)
        
    elif do == 'wavelet':
        time_find,mean_peak_power,Duration,peak_time,peak_at=spindle_validation_with_sleep_stage_after_wavelet_transform(raw,
                                                                                                     channelList,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.5,
                                                                                                 h_bound=3.0,tol=1,
                                                                                                 higher_threshold=higher_threshold)
    
    ###Taking out the first 100 seconds and the last 300 seconds###        
    result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
    result['Annotation'] = 'auto spindle'
    result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
    result = result[result.Onset > front]


    # make gold standard data frame of annotations and take out the annotations before first 300 and after the last 100 seconds
    gold_standard = read_annotation(raw,annotation_file)
    # make true labels based on gold standard annotation
    manual_labels = discritized_onset_label_manual(raw,gold_standard,3)
    # make predicted labels based on predicted onest and durations of the FBT model
    auto_labels,discritized_time_intervals = discritized_onset_label_auto(raw,result,3)
    temp_dictionary[sub+day]=[manual_labels,auto_labels,discritized_time_intervals]
    # sample the FM and miss cases
    comparedRsult = manual_labels - auto_labels
    sampling,labeling = sampling_FA_MISS_CR(comparedRsult,manual_labels, raw, annotation, 
                                            discritized_time_intervals,sampling,labeling,front=300,back=100,)
        

    return temp_dictionary,sampling,labeling

from sklearn import metrics
from collections import Counter
def fit_data(raw,exported_pipeline,annotation_file,cv,front=300,back=100,few=False):
    """
    Wrapper function for machine learning models to fit for individual EEG recording
    
    Raw: EEG raw object
    exported_pipeline: scit-kit learn machine estimator or pipeline estimator
    annotation_file: file name of the annotation file
    cv: cross validation method
    front: first # seconds of EEG recording being taken out
    back: last # seconds of EEG recording being taken out
    few: if a recording has too few true spindle, we fit the model with other data, and predict the spindle for this data
        and cross validate locally
    
    """
    data=[];
    stop = raw.times[-1]-back
    events = mne.make_fixed_length_events(raw,1,start=front,stop=stop,duration=3,)
    epochs = mne.Epochs(raw,events,1,tmin=0,tmax=3,proj=False,preload=True)
    epochs.resample(64)
    psds, freqs=mne.time_frequency.psd_multitaper(epochs,tmin=0,tmax=3,fmin=11,fmax=16,low_bias=True,proj=False,)
    psds = 10* np.log10(psds)
    data = epochs.get_data()[:,:,:-1];freqs = freqs[psds.argmax(2)];psds = psds.max(2); 
    n_ = len(raw.ch_names)
    freqs = freqs.reshape(len(freqs),n_,1);psds = psds.reshape(len(psds),n_,1)
    data = np.concatenate([data,psds,freqs],axis=2)
    data = data.reshape(len(events),-1)
    gold_standard = read_annotation(raw,annotation_file)
    manual_labels,_ = discritized_onset_label_manual(raw,gold_standard,3)
    
    

    if few:
        print('too few spindle for fiting')
        fpr,tpr=[],[];AUC=[];confM=[];sensitivity=[];specificity=[]
        #cv = KFold(n_splits=10,random_state=123345,shuffle=True)
        for ii in range(5):
            test = np.random.choice(np.arange(len(manual_labels)),size=int(len(manual_labels)*0.1),replace=False)
            while sum(manual_labels[test]) < 1:
                test = np.random.choice(np.arange(len(manual_labels)),size=int(len(manual_labels)*0.1),replace=False)
            ratio_threshold = list(Counter(manual_labels[test]).values())[1]/(list(Counter(manual_labels[test]).values())[0]+list(Counter(manual_labels[test]).values())[1])
            print(ratio_threshold)
            
            #exported_pipeline.fit(data[train,:],manual_labels[train])
            fp,tp,_ = roc_curve(manual_labels[test],exported_pipeline.predict_proba(data[test])[:,1])
            confM_temp = metrics.confusion_matrix(manual_labels[test],
                                                  exported_pipeline.predict_proba(data[test])[:,1]>ratio_threshold)
            print('confusion matrix\n',confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis])
            TN,FP,FN,TP = confM_temp.flatten()
            sensitivity_ = TP / (TP+FN)
            specificity_ = TN / (TN + FP)
            AUC.append(roc_auc_score(manual_labels[test],
                      exported_pipeline.predict_proba(data[test])[:,1]))
            fpr.append(fp);tpr.append(tp)
            confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
            confM.append(confM_temp.flatten())
            sensitivity.append(sensitivity_)
            specificity.append(specificity_)
        print(metrics.classification_report(manual_labels[test],exported_pipeline.predict_proba(data[test])[:,1]>ratio_threshold))
        return AUC,fpr,tpr,confM,sensitivity,specificity
    else:
        
        print('doing fit prediction')
        fpr,tpr=[],[];AUC=[];confM=[];sensitivity=[];specificity=[]
        for train, test in cv.split(data,manual_labels):
            C = np.array(list(dict(Counter(manual_labels[train])).values()))
            ratio_threshold = C.min() / C.sum()
            print(ratio_threshold)
            exported_pipeline.fit(data[train,:],manual_labels[train])
            fp,tp,_ = metrics.roc_curve(manual_labels[test],exported_pipeline.predict_proba(data[test])[:,1])
            confM_temp = metrics.confusion_matrix(manual_labels[test],
                                                  exported_pipeline.predict_proba(data[test])[:,1]>ratio_threshold)
            print('confusion matrix\n',confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis])
            TN,FP,FN,TP = confM_temp.flatten()
            sensitivity_ = TP / (TP+FN)
            specificity_ = TN / (TN + FP)
            AUC.append(roc_auc_score(manual_labels[test],
                      exported_pipeline.predict_proba(data[test])[:,1]))
            fpr.append(fp);tpr.append(tp)
            confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
            confM.append(confM_temp.flatten())
            sensitivity.append(sensitivity_)
            specificity.append(specificity_)
        print(metrics.classification_report(manual_labels[test],
                                            exported_pipeline.predict_proba(data[test])[:,1]>ratio_threshold))
        return AUC,fpr,tpr,confM,sensitivity,specificity
    
    return AUC,fpr,tpr
def compute_measures(dictionary_data, label='without',plot_flag=False,n_folds=10):
    random.seed(12345)
    df_accuracy=[];df_confusion_matrix=[];df_fpr=[];df_tpr=[];df_AUC=[];
    thresholds = np.sort(list(dictionary_data[label].keys()))
    for threshold in thresholds:
        temp_data = dictionary_data[label][threshold]
        manu_scores=[];auto_scores=[]
        for sub,data in temp_data.items():
            manu,auto,time_intervals = data
            manu_scores.append(manu)
            auto_scores.append(auto)

        # shuffle
        manu_scores = np.concatenate(manu_scores)
        auto_scores = np.concatenate(auto_scores)


        kf = KFold(n_splits=n_folds,random_state=12345,shuffle=True)

        # here are the measures of performance:
        temp_accuracy=[];temp_confusion_matrix = [];
        temp_fpr=[];temp_tpr=[];temp_AUC=[];
        ### end of measurs ###

        for train_index, test_index in kf.split(auto_scores,manu_scores):
            temp_manu = manu_scores[train_index]
            temp_auto = auto_scores[train_index]
            temp_accuracy.append(accuracy_score(temp_manu,temp_auto))
            temp_confusion_matrix.append(confusion_matrix(temp_manu,temp_auto))
            fpr,tpr,T = roc_curve(temp_manu,temp_auto)
            temp_fpr.append(fpr)
            temp_tpr.append(tpr)
            temp_AUC.append(roc_auc_score(temp_manu,temp_auto))
        ### save measures ###
        df_accuracy.append(temp_accuracy)
        df_confusion_matrix.append(temp_confusion_matrix)
        df_fpr.append(temp_fpr)
        df_tpr.append(temp_tpr)
        df_AUC.append(temp_AUC)
        
        
    if plot_flag:
        df_plot={}
        df_plot['accuracy']=np.array(df_accuracy)
        df_plot['confusion_matrix']=np.array(df_confusion_matrix)
        df_plot['fpr']=np.array(df_fpr)
        df_plot['tpr']=np.array(df_tpr)
        df_plot['AUC']=np.array(df_AUC)
        df_plot['thresholds']=np.array(thresholds)
        return df_plot
    else:
        return df_accuracy,df_confusion_matrix,df_fpr,df_tpr,df_AUC,thresholds
        
def compute_two_thresholds(dictionary_data, label='without',plot_flag=False,n_folds=10):
    random.seed(12345)
    threshold_list=[]
    df_accuracy=[];df_confusion_matrix=[];df_fpr=[];df_tpr=[];df_AUC=[];
    dictionary_data=dictionary_data[label]
    for key in dictionary_data.keys():
        current_data = dictionary_data[key]
        lower, upper = key.split(',')

        manu_scores=[];auto_scores=[]
        for sub,data in current_data.items():
            manu,auto,time_intervals = data
            manu_scores.append(manu)
            auto_scores.append(auto)
    
        # shuffle
        manu_scores = np.concatenate(manu_scores)
        auto_scores = np.concatenate(auto_scores)
    
    
        kf = KFold(n_splits=n_folds,random_state=12345,shuffle=True)
    
        # here are the measures of performance:
        temp_accuracy=[];temp_confusion_matrix = [];
        temp_fpr=[];temp_tpr=[];temp_AUC=[];
        ### end of measurs ###
    
        for train_index, test_index in kf.split(auto_scores,manu_scores):
            temp_manu = manu_scores[train_index]
            temp_auto = auto_scores[train_index]
            temp_accuracy.append(accuracy_score(temp_manu,temp_auto))
            temp_confusion_matrix.append(confusion_matrix(temp_manu,temp_auto))
            fpr,tpr,T = roc_curve(temp_manu,temp_auto)
            temp_fpr.append(fpr)
            temp_tpr.append(tpr)
            temp_AUC.append(roc_auc_score(temp_manu,temp_auto))
        ### save measures ###
        df_accuracy.append(temp_accuracy)
        df_confusion_matrix.append(temp_confusion_matrix)
        df_fpr.append(temp_fpr)
        df_tpr.append(temp_tpr)
        df_AUC.append(temp_AUC)
        
        threshold_list.append([float(lower),float(upper),np.mean(temp_AUC),np.std(temp_AUC),
                              np.mean(temp_accuracy),np.std(temp_accuracy)])
        
        
        result = pd.DataFrame(threshold_list,columns=['lower_threshold','upper_threshold','mean_AUC','std_AUC',
                                              'mean_accuracy','std_accuracy'])

        result = result.sort_values(['lower_threshold','upper_threshold'])
        result = result.reset_index(drop=True)
    if plot_flag:
        df_plot={}
        df_plot['accuracy']=np.array(df_accuracy)
        df_plot['confusion_matrix']=np.array(df_confusion_matrix)
        df_plot['fpr']=np.array(df_fpr)
        df_plot['tpr']=np.array(df_tpr)
        df_plot['AUC']=np.array(df_AUC)
        #df_plot['thresholds']=np.array(thresholds)
        return df_plot
    else:
        return df_accuracy,df_confusion_matrix,df_fpr,df_tpr,df_AUC,threshold_list,result
        
def detection_pipeline_crossvalidation(raw,channelList,annotation,windowSize,lower_threshold,higher_threshold,syn_channel,l,h,annotation_file,cv=None,front=300,back=100,auc_threshold=0.5):
    time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=thresholding_filterbased_spindle_searching(raw,channelList,annotation,moving_window_size=windowSize,
                                                                                                    lower_threshold=lower_threshold,
                                        syn_channels=3,l_bound=0.5,h_bound=2,tol=1,higher_threshold=higher_threshold,
                                        front=300,back=100,sleep_stage=True,proba=True,validation_windowsize=3
                                        )
    """
    Wrapper function for the FBT model to cross validate with individual recording
    
    raw: raw EEG recording object
    channelList: channel of interest
    annotation: data frame of spindle, sleep stage annotation
    windowSize: covolution RMS computing window size
    lower_threshold: lower threshold
    higher_threshold: higher_threshold
    syn_channel: # of channels should be agree with each other
    l: low cutoff frequency
    h: high cutoff frequncy
    annotation_file: file name of the annotations, in .txt
    cv: cross validation method
    front:
    back:
    auc_threshold: decision making boundary
    
    """

    #anno = annotation[annotation.Annotation == 'spindle']['Onset']
    gold_standard = read_annotation(raw,annotation_file)
    manual_labels,_ = discritized_onset_label_manual(raw,gold_standard,3)
    
                                                                                               
    raw.close()  
    temp_auc = [];#fp=[];tp=[]
    confM = [];sensitivity=[];specificity=[];fpr=[];tpr=[]
    if cv == None:
        cv = KFold(n_splits=10,random_state=12345,shuffle=True)
    if auc_threshold == 0.5:
        for train, test in cv.split(manual_labels):
            detected,truth,detected_proba = auto_label[train],manual_labels[train],auto_proba[train]
            temp_auc.append(roc_auc_score(truth,detected))
            confM_temp = metrics.confusion_matrix(truth,detected)
            TN,FP,FN,TP = confM_temp.flatten()
            sensitivity_ = TP / (TP+FN)
            specificity_ = TN / (TN + FP)
            confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
            print(confM_temp)
            confM.append(confM_temp.flatten())
            sensitivity.append(sensitivity_)
            specificity.append(specificity_)
            fp,tp,t = roc_curve(truth,detected_proba)
            fpr.append(fp)
            tpr.append(tp)
        
        print(metrics.classification_report(manual_labels,auto_label))
        return temp_auc,fpr,tpr, confM, sensitivity, specificity
    else:
        auc_threshold = list(Counter(auto_label).values())[1]/(list(Counter(auto_label).values())[0]+list(Counter(auto_label).values())[1])
        for train, test in cv.split(manual_labels):
            truth,detected_proba = manual_labels[train],auto_proba[train]
            detected = detected_proba > auc_threshold
            temp_auc.append(roc_auc_score(truth,detected_proba))
            confM_temp = metrics.confusion_matrix(truth,detected)
            TN,FP,FN,TP = confM_temp.flatten()
            sensitivity_ = TP / (TP+FN)
            specificity_ = TN / (TN + FP)
            confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
            print(confM_temp)
            confM.append(confM_temp.flatten())
            sensitivity.append(sensitivity_)
            specificity.append(specificity_)
            fp,tp,t = roc_curve(truth,detected_proba)
            fpr.append(fp)
            tpr.append(tp)
        
        print(metrics.classification_report(manual_labels,auto_proba>auc_threshold))
        return temp_auc,fpr,tpr, confM, sensitivity, specificity
from random import shuffle
from scipy.stats import percentileofscore
def Permutation_test_(data1, data2, n1=100,n2=100):
    p_values = []
    for simulation_time in range(n1):
        shuffle_difference =[]
        experiment_difference = np.mean(data1,0) - np.mean(data2,0)
        vector_concat = np.concatenate([data1,data2])
        for shuffle_time in range(n2):
            shuffle(vector_concat)
            new_data1 = vector_concat[:len(data1)]
            new_data2 = vector_concat[len(data1):]
            shuffle_difference.append(np.mean(new_data1) - np.mean(new_data2))
        p_values.append(min(percentileofscore(shuffle_difference,experiment_difference)/100,
                            (100-percentileofscore(shuffle_difference,experiment_difference))/100))
    
    return p_values,np.mean(p_values),np.std(p_values)
from sklearn.model_selection import permutation_test_score,StratifiedKFold
from sklearn import utils
def Permutation_test(data,n_permutations=100,n_=100):
    data = np.array(data)
    p_vals = []
    temp_df = {}
    for ii,d in enumerate(data):
        d = np.array(d)
        temp_df[ii] = d
        temp_df['label%d'%ii] = np.ones(d.shape) * ii
        
    vectorized_data = np.concatenate([temp_df[ii] for ii in range(data.shape[0])])
    labels  = np.concatenate([temp_df['label%d'%ii] for ii in range(data.shape[0])])
    
    for iiii in range(100):
        vectorized_data,labels = utils.shuffle(vectorized_data,labels)
    
    for simu in tqdm(range(n_)):
        cv = StratifiedKFold(n_splits=5,shuffle=True)#,random_state=12345)
        clf = LogisticRegressionCV(Cs=np.logspace(-3,3,7),cv=3,)#random_state=12345,)
        score,permutation_scores,pval = permutation_test_score(clf,vectorized_data.reshape(-1,1),labels,cv=cv,
                                                               n_permutations=n_permutations,)#random_state=12345)
        p_vals.append(pval)
    return score,p_vals,np.mean(p_vals),np.std(p_vals)