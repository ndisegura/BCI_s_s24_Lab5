# -*- coding: utf-8 -*-
"""
@author: Andres Segura & Lute Lillo Portero
BME 6770: BCI's Lab 05
Dr. David Jangraw
4/13/2024

UPDATE Module Description:


This module provides functions to process ...(e.g Steady State Visual Evoked Potential signals.
The module will analyze the SSVEP data, generate predicted labels, compute accuracy and
information tranfer rate. The module also implements function to plot a confusion matrix
for accuracy and ITRgenerates, and predictor histograms.)
"""

# Import Statements
from pylab import *
from scipy.signal import firwin, filtfilt, freqz, hilbert
import matplotlib.pyplot as plt
import numpy as np
import import_ssvep_data as imp



def load_data(data_directory,channels_to_plot=None):
    '''
    This function loads audiovis data and stores it in a dictionary for easy 
    access.
    
    Parameters
    ----------
    
    data_directory :str
        path to data folder.
    channels_to_plot: list of strings
        Optional. List of channel to plot after importing the data.
    
    
    Returns
    -------
    data_dict : dictonary 
         containing feilds of AudioVis data representing strings, floats, and bool, 
         size of dictionary 1 x N where N is the amount of fields
    
    '''
    data_file=f'{data_directory}AudVisData.npy'
    
    # Load dictionary
    data_dict = np.load(data_file,allow_pickle=True).item()
#    data_dict = np.load(data_file,allow_pickle=True)

    if channels_to_plot: 
        pass
    
    
    return data_dict

