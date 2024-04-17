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
import matplotlib.pyplot as plt
import numpy as np
import plot_topo



def load_data(data_directory, channels_to_plot=None):
    """
    Load EEG data from the specified directory.

    Args:
        data_directory (str): Path to the directory containing EEG data.
        channels_to_plot (list or None, optional): List of channel names to plot. Defaults to None.

    Returns:
        data_dict: A dictionary containing EEG data.

    """
    data_file=f'{data_directory}AudVisData.npy'
    
    # Load dictionary
    data_dict = np.load(data_file,allow_pickle=True).item()

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
            axs[channel_index].set_ylabel(f'Voltage on {channel_name} ({units})\n')
            axs[channel_index].set_xlim(54,61)
    
        plt.tight_layout()  
        plt.savefig(f"plots/Raw_AudioVis.png")  
    
    return data_dict

def plot_components(mixing_matrix, channels, components_to_plot):
    """
    Plot independent components obtained from ICA.

    Args:
        mixing_matrix (numpy.ndarray): Mixing matrix obtained from ICA of shape (C,C)
            where C is the number of channels.
        channels (list): List of channel names.
        components_to_plot (list): List of indices of components to plot.

    Returns:
        None

    """
    plot_count = len(components_to_plot)
    plt.figure(figsize=(15, 8))
    for component_index, component_value in enumerate(components_to_plot):
        
        if plot_count <= 5:
            subplot(1,5,component_index+1)
            my_component = mixing_matrix[:,component_value]
            img, cbar = plot_topo.plot_topo(channel_names=channels, channel_data=my_component,
                                title=f'ICA component {component_index}', cbar_label='', montage_name='standard_1005')
            
        if plot_count <= 10:
            subplot(2, 5, component_index+1)
            my_component=mixing_matrix[:,component_value]
            plot_topo.plot_topo(channel_names=list(channels), channel_data=my_component, title=f'ICA component {component_index}',cbar_label='', montage_name='standard_1005')
    
    plt.tight_layout()
    plt.savefig(f"plots/components_scalp.png")
    plt.close()
    
# Part 3
def get_sources(eeg, unmixing_data, fs, sources_to_plot):
    """
    Project EEG data into ICA source space.

    Args:
        eeg (numpy.ndarray): EEG data of shape (N, D), where N is the number of channels and D is the number of samples.
        unmixing_data (numpy.ndarray): Unmixing matrix obtained from ICA.
        fs (float): Sampling frequency.
        sources_to_plot (list or None, optional): List of indices of sources to plot. Defaults to None.

    Returns:
        source_activations (numpy.ndarray): Source activations in ICA space of shape (N, D).

    """
    eeg_time = np.arange(0,len(eeg[0])*1/fs,1/fs)
    source_activations = np.matmul(unmixing_data, eeg)
    
    if sources_to_plot:
        plot_count=len(sources_to_plot)
        fig,axs=plt.subplots(nrows=plot_count,ncols=1,sharex=True)
        fig.suptitle(' AudioVis EEG Data in ICA Source Space', fontsize=18)
        
        for channel_index, source_int in enumerate(sources_to_plot):
        
            axs[channel_index].plot(eeg_time, source_activations[source_int, :], label=source_int)
            axs[channel_index].set_xlabel('Eeg time (s)')
            axs[channel_index].set_ylabel(f'Source {source_int} (uV)\n')
            axs[channel_index].set_xlim(54, 61)
    
        plt.tight_layout()    
        plt.savefig(f"plots/AudioVis_EEG_Data_Source_Space.png")
        plt.close()
        
    return source_activations

# Part 4 - 
def remove_sources(source_activations, mixing_matrix, sources_to_remove):
    """
    Remove specific sources from the source activations.

    Args:
        source_activations (numpy.ndarray): Source activations in ICA space of shape (N, D), 
        where N is the number of channels and D is the number of samples..
        mixing_matrix (numpy.ndarray): Mixing matrix obtained from ICA of shape (N, N)
            where N is the number of channels
        sources_to_remove (list or None, optional): List of indices of sources to remove. Defaults to None.

    Returns:
        cleaned_eeg (numpy.ndarray): Cleaned EEG data of shape (N, D).

    """
    # Zero-out specific sources
    if sources_to_remove:
        for source in sources_to_remove:
            source_activations[source, :] = 0
        
    # Transform back into electrode space
    cleaned_eeg = np.matmul(mixing_matrix, source_activations)
    
    return cleaned_eeg

# Part 5 - Compare reconstructions
def compare_reconstructions(eeg, reconstructed_eeg, cleaned_eeg, fs, channels, channels_to_plot):
    """
    Compare raw EEG data, reconstructed EEG data, and cleaned EEG data.

    Args:
        eeg (numpy.ndarray): Raw EEG data of shape (N, D), where N is the number of channels and D is the number of samples.
        reconstructed_eeg (numpy.ndarray): Reconstructed EEG data of shape (N, D).
        cleaned_eeg (numpy.ndarray): Cleaned EEG data of shape (N, D).
        fs (float): Sampling frequency.
        channels (list): List of channel names.
        channels_to_plot (list): List of channel names to plot.

    Returns:
        None

    """
    eeg_time = np.arange(0,len(eeg[0])*1/fs,1/fs)
    
    plot_count = len(channels_to_plot)
    fig,axs = plt.subplots(nrows=plot_count, ncols=1, sharex=True)
    fig.suptitle(' AudioVis EEG Data Reconstructed and Clean', fontsize=18)

    for channel_index, channel_name in enumerate(channels_to_plot):
        axs[channel_index].plot(eeg_time, np.squeeze(eeg[channels==channel_name]), label='raw', color='blue')
        axs[channel_index].plot(eeg_time, np.squeeze(reconstructed_eeg[channels==channel_name]), label='reconstructed', color='green', linestyle='dashed')
        axs[channel_index].plot(eeg_time, np.squeeze(cleaned_eeg[channels==channel_name]), label='cleaned', color='red', linestyle='dotted')
        axs[channel_index].set_xlabel('Eeg time (s)')
        axs[channel_index].set_ylabel(f'Voltage on {channel_name} (uV)\n')
        axs[channel_index].set_xlim(55,60)
        axs[channel_index].legend()
        
    plt.tight_layout()
    plt.savefig(f"plots/comparison_eeg.png")