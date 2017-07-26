# Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization
Scripts and figures that included in the paper


1. [Figure](https://github.com/adowaconan/Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization/tree/master/Figures) folder contains the final figures that used in the paper
2. [annotations](https://github.com/adowaconan/Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization/tree/master/annotations) folder contains the manually marked sleep spindles, and also k-complexes and sleep stages
3. [eegPinelineDesign.py](https://github.com/adowaconan/Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization/blob/master/eegPinelineDesign.py) (I know, I can't spell) is the script contains all the functions I have ever used during this project
4. [Filter_based_and_thresholding.py](https://github.com/adowaconan/Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization/blob/master/Filter_based_and_thresholding.py) is the main function of the model
5. [threshold_optimization.py](https://github.com/adowaconan/Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization/blob/master/threshold_optimization.py) is the script you can use to optimize the lower and higher thresholds for one dataset
6. In the paper, we used permutation test to compare classification results pair-wise. Thus, we publish the [code](https://github.com/adowaconan/Identifying_sleep_spindles_with_multichannel_EEG_and_classification_optimization/blob/master/permutation%20test%20used%20in%20the%20current%20study.py) that is used for the test. 
