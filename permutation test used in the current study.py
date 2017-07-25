# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:01:14 2017

@author: ning
"""

import numpy as np
from random import shuffle
from scipy.stats import percentileofscore

def Permutation_test(data1, data2):
    """
    The permutation test was used to conduct non parametric statistical analysis
    to compare two groups of data. In the current study, it was used to compare the 
    spindle detection between day1 dataset and day2 dataset and the cross validation 
    performance of filter based and thresholding model and the selected machine 
    learning model.
    
    We first compute the difference of dataset1 and dataset2. A concatenated vector
    containing dataset1 and dataset2 is constructed. During the shuffle processes (N=500),
    each time we shuffle the concatenated vector, we generate new dataset1 and new 
    dataset2. We compute the difference of the two new datasets. The differences 
    generated during the shuffle process will form a distribution of the shuffling
    results. A p value is computed by determining wheere the original difference between 
    the original dataset1 and dataset2 locates in the distribution of the shuffling results.
    Repeat this process many more times (N=500), and we sample a distribution of p values.
    
    A full comparison between parametric t test and permutation test is available:
        https://github.com/adowaconan/parametric_nonparametric_statistics_comparison/blob
        /master/t%20tests/two%20sample%20t%20test.ipynb
    inputs:
        data1: numpy array or list of data
        data2: numpy array or list of data
    
    return:
        
        1. list of p values computed during the shuffle process
        2. average of p values
        3. standard deviation of the p values
    """
    p_values = []
    for simulation_time in range(500):
        shuffle_difference =[]
        experiment_difference = np.mean(data1,0) - np.mean(data2,0)
        vector_concat = np.concatenate([data1,data2])
        for shuffle_time in range(500):
            shuffle(vector_concat)
            new_data1 = vector_concat[:len(data1)]
            new_data2 = vector_concat[len(data1):]
            shuffle_difference.append(np.mean(new_data1) - np.mean(new_data2))
        p_values.append(min(percentileofscore(shuffle_difference,experiment_difference)/100,
                            (100-percentileofscore(shuffle_difference,experiment_difference))/100))
    
    return p_values,np.mean(p_values),np.std(p_values)