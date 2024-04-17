# -*- coding: utf-8 -*-
"""
Andres Segura & Aiden Lute Lillo Portero
BME 6770: BCI's Project 02
Dr. David Jangraw
4/13/2024

Test scrip for the implementation of the remove_audiovis_blinks module.
This script consist of 5 major sections. 
The test module will load the Audio Visual data into a pickable data dictionary
Using the NME library wrapper, ICA components are ploted for all the EEG channels
Addional function can obtain data from component to source space
The last function will compare the raw, cleaned, and reconstructed eeg data.
"""

# Import the necessary modules
import os
import sys
import matplotlib.pyplot as plt




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

"""
It appears a blink artifact occurrs at t=57.1s. It is more pronounced on channel FPz 
and sutle on the other two channels, perhaps convolved with other artifacts
"""

#%%
# Part 2 Plot ICA Components
eeg = data['eeg']
fs = data['fs']
channels = data['channels']
unmixing_matrix = data['unmixing_matrix']
mixing_matrix = data['mixing_matrix']

components_to_plot = [0,1,2,3,4,5,6,7,8,9]

#Plot ICA components
rmv.plot_components(mixing_matrix,channels,components_to_plot) 

"""
Components 0 and 8 appear to be ocular artifacts due to the increased shading in the 
frontal area. Component 9 also exhibits EOG artifacts characteristics as lateral eye movemtns
due to the increased shading in the topo map closer to the temples
""" 

#%%
# Part 3 - Source Activity
sources_to_plot = [0, 8, 4]
source_activations = rmv.get_sources(eeg, unmixing_matrix, fs, sources_to_plot)
#%%
# Part 4 - Remove Sources. Make two calls. One to clean-out sources identified and one without.
# Sources to be removed 1 -> Fpz, 29 -> Cz
sources_to_remove = [0,8]
cleaned_eeg = rmv.remove_sources(source_activations, mixing_matrix, sources_to_remove)

sources_to_remove = []
reconstructed_eeg = rmv.remove_sources(source_activations, mixing_matrix, sources_to_remove)
#%%
# Part 5 - Compare reconstructions
rmv.compare_reconstructions(eeg, reconstructed_eeg, cleaned_eeg, fs, channels, channels_to_plot)

"""
By comparisson we can see that artifacts are mostly removed from channel "Fpz", but interestely
reconstructed data also seems to be affected equally. The transformation of the mixing and unmixing 
matrix in this case did not results in a accurate representation of the original signals. 
Channels Cz also experienced removal of some of the spike artifacs.
The last channel had the least effect of artifact removal from  ICA components. 

Without looking at the comparisson plot, the electrodes that would be mostly affected by removing artifacts
are the ones spacially located closer to the activation source.



"""