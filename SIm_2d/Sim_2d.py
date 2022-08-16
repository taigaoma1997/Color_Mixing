# -*- coding: utf-8 -*-
"""
Created on Mon Aug 5 21:23:28 2022

@author: mtg
"""

import matplotlib
import matplotlib.pyplot as plt
from tmm import coh_tmm, inc_tmm
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import pandas as pd
import colour
import matplotlib as mpl
from colour.plotting import plot_chromaticity_diagram_CIE1931, ColourSwatch, plot_multi_colour_swatches, plot_sds_in_chromaticity_diagram_CIE1931
from colour.plotting.models import plot_RGB_colourspaces_in_chromaticity_diagram
from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution

from utility import *

DATABASE = './data'
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 12
#mpl.rcParams['axes.titlepad'] = -4
mpl.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


################################################################################################################
# Please edit the following variables

file_name = 'example_1d'   # Specify the saved file name, file will be saved in the same folder
 
structure = ['Ti3O5', 'Cr', 'Ti3O5', 'Cu', 'Ti3O5', 'Cr', 'Ti3O5'] # The thin film is recorded from top to bottom

thickness = [0, 11, 100, 70, 200, 11, 35]  # specify the thickness at each thin film layer. 0 means scanning the thickness

num_scan = [i for i in range(len(thickness)) if thickness[i] == 0]  # Examine the number of scanning thickness
if len(num_scan) not in [1, 2, 3]:
    raise KeyError('Please make sure there are at least 1 layer for scanning / at most 3 layers for scanning. Use 0 to specify the layer of scanning. ')

# Specify the thickness scanning range and scanning steps (all in nm)

t_scan_1 = [25, 300]
t_scan_step_1 = 25

t_scan_2 = [25, 300]
t_scan_step_2 = 25

t_scan_3 = [25, 100]
t_scan_step_3 = 25

if t_scan_1[0] >= t_scan_1[1] or t_scan_2[0] >= t_scan_2[1]  or t_scan_3[0] >= t_scan_3[1] :
    raise KeyError('Please make sure the scanning range is reasonable.')

# End of edits. Please do not change the following codes. 
################################################################################################################

# Set up the wavelength range (in um)

lamda_low = 0.4    
lamda_high = 0.8
wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.005)  # ranges from 400 to 800 nm

# Load all related materials. if there are new materials, please add here

mats = ['Al', 'Al2O3', 'Cr', 'Cu', 'Fe2O3', 'Glass', 'SiO2', 'Ti', 'Ti3O5'] 
nk_dict = load_materials(all_mats = mats, wavelengths = wavelengths)

for i in structure:
    if i not in mats:
        raise KeyError('The input structure contains unknown materials. Please examine the input')

substrate = 'Glass'  # The substrate


if len(num_scan) ==1:
    # calculating 1d scanning
    
    all_RGB = scan_1D(num_scan, t_scan_1, t_scan_step_1, structure, thickness, wavelengths, nk_dict, substrate)
    
    fig = plt.figure(dpi=200)
    fig.patch.set_facecolor('white')
    plt.imshow(np.clip(np.array(all_RGB), 0, 1))
    
    x_ticks = [str(i) for i in range(t_scan_1[0], t_scan_1[1]+1, t_scan_step_1)]
    plt.xticks(range(0, len(all_RGB[0]), 1), x_ticks, fontsize=8,  rotation =45)
    #plt.tick_params(left = False, bottom = True)
    plt.yticks([])
    
    plt.axvline(x=-0.5,color='black', linewidth=1)
    plt.axhline(y=-0.5,color='black', linewidth=1)
    for i in range(len(all_RGB[0])):
        plt.axvline(x=i + 0.5,color='black', linewidth=1)
    for i in range(len(all_RGB)):
        plt.axhline(y=i + 0.5,color='black', linewidth=1)
    plt.xlabel('Layer ' + str(num_scan[0]+1) + ' : '+structure[num_scan[0]]+' thickness (nm)', fontsize=8)
    titles = ''
    titles_1 = ''
    for i in range(len(structure)):
        titles += structure[i] 
        titles += '    '
        if i not in num_scan:
            titles_1 += str(int(thickness[i]))
            titles_1 += 'nm  '
        else:
            titles_1 += ' ? nm  '
            
    plt.title(titles+'\n'+titles_1, fontsize=8)
    plt.savefig(file_name+'.jpg')

elif len(num_scan)== 2:
    # calculating 2D scanning
    
    all_RGB = scan_2D(num_scan, t_scan_1, t_scan_2, t_scan_step_1, t_scan_step_2, structure, thickness, wavelengths, nk_dict, substrate)
    fig = plt.figure(dpi=200)
    fig.patch.set_facecolor('white')
    plt.imshow(np.clip(np.array(all_RGB), 0, 1))
    
    y_ticks = [str(i) for i in range(t_scan_2[0], t_scan_2[1]+1, t_scan_step_2)]
    plt.yticks(range(0, len(all_RGB), 1), y_ticks[::-1], fontsize=8)
    
    x_ticks = [str(i) for i in range(t_scan_1[0], t_scan_1[1]+1, t_scan_step_1)]
    plt.xticks(range(0, len(all_RGB[0]), 1), x_ticks, fontsize=8,  rotation =45)
    
    plt.axvline(x=-0.5,color='black', linewidth=1)
    plt.axhline(y=-0.5,color='black', linewidth=1)
    for i in range(len(all_RGB[0])):
        plt.axvline(x=i + 0.5,color='black', linewidth=1)
    for i in range(len(all_RGB)):
        plt.axhline(y=i + 0.5,color='black', linewidth=1)
    plt.xlabel('Layer ' + str(num_scan[0]+1) + ' : '+structure[num_scan[0]]+' thickness (nm)', fontsize=8)
    plt.ylabel('Layer ' + str(num_scan[1]+1) + ' : '+structure[num_scan[1]]+' thickness (nm)', fontsize=8)
    titles = ''
    titles_1 = ''
    for i in range(len(structure)):
        titles += structure[i] 
        titles += '    '
        if i not in num_scan:
            titles_1 += str(int(thickness[i]))
            titles_1 += 'nm  '
        else:
            titles_1 += ' ? nm  '
            
    plt.title(titles+'\n'+titles_1, fontsize=8)
    plt.savefig(file_name+'.jpg')

else:
    # calculating 3D scanning
    
    for scan_d in range(t_scan_3[0], t_scan_3[1], t_scan_step_3):
        
        
        thickness[num_scan[2]] = scan_d
        
        
        all_RGB = scan_2D(num_scan, t_scan_1, t_scan_2, t_scan_step_1, t_scan_step_2, structure, thickness, wavelengths, nk_dict, substrate)
        fig = plt.figure(dpi=200)
        fig.patch.set_facecolor('white')
        plt.imshow(np.clip(np.array(all_RGB), 0, 1))
        
        y_ticks = [str(i) for i in range(t_scan_2[0], t_scan_2[1]+1, t_scan_step_2)]
        plt.yticks(range(0, len(all_RGB), 1), y_ticks[::-1], fontsize=8)
        
        x_ticks = [str(i) for i in range(t_scan_1[0], t_scan_1[1]+1, t_scan_step_1)]
        plt.xticks(range(0, len(all_RGB[0]), 1), x_ticks, fontsize=8,  rotation =45)
        
        plt.axvline(x=-0.5,color='black', linewidth=1)
        plt.axhline(y=-0.5,color='black', linewidth=1)
        for i in range(len(all_RGB[0])):
            plt.axvline(x=i + 0.5,color='black', linewidth=1)
        for i in range(len(all_RGB)):
            plt.axhline(y=i + 0.5,color='black', linewidth=1)
        plt.xlabel('Layer ' + str(num_scan[0]+1) + ' : '+structure[num_scan[0]]+' thickness (nm)', fontsize=8)
        plt.ylabel('Layer ' + str(num_scan[1]+1) + ' : '+structure[num_scan[1]]+' thickness (nm)', fontsize=8)
        titles = ''
        titles_1 = ''
        for i in range(len(structure)):
            titles += structure[i] 
            titles += '    '
            if i not in num_scan[:2]:
                titles_1 += str(int(thickness[i]))
                titles_1 += 'nm  '
            else:
                titles_1 += ' ? nm  '
                
        plt.title('Layer '+str(num_scan[2])+' : '+structure[num_scan[2]]+ ' thickness = '+str(scan_d)+'nm\n'+titles+'\n'+titles_1, fontsize=8)
        plt.savefig(file_name+'_layer_'+str(num_scan[2])+'_'+str(scan_d)+'nm.jpg')
    



    
    


















