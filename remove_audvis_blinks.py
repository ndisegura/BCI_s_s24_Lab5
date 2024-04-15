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
import plot_topo



def load_data(data_directory, channels_to_plot=None):
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
        eeg=data_dict['eeg']
        channels=data_dict['channels']
        fs=data_dict['fs']
        eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)
        units=data_dict['units']
        plot_count=len(channels_to_plot)
        fig,axs=plt.subplots(nrows=plot_count, ncols=1, sharex=True)
        fig.suptitle(' Raw AudioVis EEG Data ', fontsize=18)
 
        for channel_index,channel_name in enumerate(channels_to_plot):
            axs[channel_index].plot(eeg_time, np.squeeze(eeg[channels==channel_name]),label=channel_name)
            axs[channel_index].set_xlabel('Eeg time (s)')
            axs[channel_index].set_ylabel(f'Voltage on {channel_name}\n ({units})')
            axs[channel_index].set_xlim(54,61)
    
        # plt.tight_layout()  
        plt.savefig(f"plots/Raw_AudioVis.png")  
    
    return data_dict

def plot_components(mixing_matrix, channels, components_to_plot):
    
    plot_count = len(components_to_plot)
    
    for component_index, component_value in enumerate(components_to_plot):
        
        if plot_count <= 5:
            subplot(1,5,component_index+1)
            my_component = mixing_matrix[:,component_value]
            img, cbar = plot_topo.plot_topo(channel_names=channels, channel_data=my_component,
                                title=f'ICA component {component_index}', cbar_label='', montage_name='standard_1005')
            plt.savefig(f"plots/scalp_figures/{component_index}_scalp.png")
            
        if plot_count <= 10:
            
            subplot(2, 5, component_index+1)
            my_component=mixing_matrix[:,component_value]
            plot_topo.plot_topo(channel_names=list(channels), channel_data=my_component, title=f'ICA component {component_index}',cbar_label='', montage_name='standard_1005')
            plt.savefig(f"plots/scalp_figures/{component_index}_scalp.png")
            
# Part 3
def get_sources(eeg, unmixing_data, fs, sources_to_plot):
    
    eeg_time = np.arange(0,len(eeg[0])*1/fs,1/fs)

    if sources_to_plot:
        plot_count=len(sources_to_plot)
        fig,axs=plt.subplots(nrows=plot_count,ncols=1,sharex=True)
        fig.suptitle(' AudioVis EEG Data in ICA Source Space', fontsize=18)
        
        for channel_index, channel_name in enumerate(sources_to_plot):
            source_activity = np.matmul(eeg.T, unmixing_data)
            
            axs[channel_index].plot(eeg_time, source_activity[:, channel_index], label=channel_name)
            axs[channel_index].set_xlabel('Eeg time (s)')
            axs[channel_index].set_ylabel(f'Voltage on {channel_name} (uV)\n')
            axs[channel_index].set_xlim(54, 61)
    
        plt.tight_layout()    
        plt.savefig(f"plots/test.png")
        
    return source_activity
        
    
    
    
