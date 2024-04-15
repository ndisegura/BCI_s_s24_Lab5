# -*- coding: utf-8 -*-
"""
Andres Segura & Aiden Lute Lillo Portero
BME 6770: BCI's Project 02
Dr. David Jangraw
4/13/2024

Test scrip for the implementation of the remove_audiovis_blinks module...(e.g.This script consist of 5 major sections.
The test module will load the SSVEP from the data dictionary and create a string array with the predicted labels.
figures of merit are then computed for different epoch start and end times)
"""

# Import the necessary modules
import os
import sys
import matplotlib.pyplot as plt
import import_ssvep_data
import filter_ssvep_data
import import_ssvep_data as imp
import predict_ssvep_data as prd
import remove_audvis_blinks as rmv

# Close previosly drawn plots
plt.close('all')

# Build data file string
data_directory = './AudVisData/'

#%%
# Part1: Import the data
# Load subject data
channels_to_plot = ['Fpz','Cz','Iz']
data = rmv.load_data(data_directory, channels_to_plot)

#%%
# Part 2 Plot ICA Components
eeg = data['eeg']
fs = data['fs']
channels = data['channels']
unmixing_matrix = data['unmixing_matrix']
mixing_matrix = data['mixing_matrix']

components_to_plot = [0,1,2,3,4,5,6,7,8,29]

# something is not righ in the lab5 instructions. assuming for now that mixing_matrix argument
# is in fact eeg raw data 
rmv.plot_components(mixing_matrix,channels,components_to_plot) 

# Part 3 - Source Activity
# rmv.get_sources(eeg, unmixing_matrix, fs, channels_to_plot)
