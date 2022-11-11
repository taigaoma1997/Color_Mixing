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

thickness = [50, 11, 100, 10, 200, 11, 35]  # specify the initial thickness at each thin film layer

layer_scan = [1, 2, 1, 2, 1, 2, 3]  # specify the layers to be scanned simultaneously. 
                                    # 0 is not scan, 1, 2, 3 means scanning
                                    # Can only be [0, 1, 2, 3] -> at most 3 different layers to scan. 
                                    # If scan, Make sure the appear order is 1 -> 2 -> 3
    
# Specify the thickness scanning range and scanning steps (all in nm)

t_scan_1 = [25, 100]
t_scan_step_1 = 25

t_scan_2 = [5, 20]
t_scan_step_2 = 5

t_scan_3 = [25, 100]
t_scan_step_3 = 25


# End of edits. Please do not change the following codes. 
################################################################################################################

# Examine if the input are 

if len(structure)!= len(thickness) or len(structure) != len(layer_scan):
    raise KeyError('Please make sure the length of structure matches the length of thickness.')

layer_scan_order = {'0':[], '1':[], '2':[], '3':[]}

for i in range(len(layer_scan)):
    
    if not isinstance(layer_scan[i], int):
        raise KeyError('Please make sure the input of layer_scan is integers.') 
    if layer_scan[i] not in [0, 1, 2, 3]:
        raise KeyError('Please make sure the input of layer_scan is only 0, 1, 2, 3.') 
        
    layer_scan_order[str(layer_scan[i])].append(i)

if (len(layer_scan_order['1']) == 0 and (len(layer_scan_order['2']) > 0 or len(layer_scan_order['2']) > 0)) or ( len(layer_scan_order['2']) == 0 and len(layer_scan_order['3']) != 0):
    raise KeyError('Please examine the order of 1, 2, 3 layers. 1 -> 2 -> 3') 
            
if len(layer_scan_order['3']) == 0:
    if len(layer_scan_order['2']) == 0:
        num_scan = 1
    else:
        num_scan = 2
else:
    num_scan = 3
    
        
if t_scan_1[0] >= t_scan_1[1] or t_scan_2[0] >= t_scan_2[1]  or t_scan_3[0] >= t_scan_3[1] :
    raise KeyError('Please make sure the thickness scanning range is reasonable.')

print('Start scanning:\n')
print('In total, there are '+str(num_scan) + ' effective layers to scan.')
scanning_name_0 = ['*'+structure[i]+'*' if i in layer_scan_order['0'] else structure[i]  for i in range(len(structure)) ]
scanning_name_1 = ['*'+structure[i]+'*' if i in layer_scan_order['1'] else structure[i]  for i in range(len(structure)) ]
scanning_name_2 = ['*'+structure[i]+'*' if i in layer_scan_order['2'] else structure[i]  for i in range(len(structure)) ]
scanning_name_3 = ['*'+structure[i]+'*' if i in layer_scan_order['3'] else structure[i]  for i in range(len(structure)) ]

print('Scanning order 0: ', scanning_name_0)
print('Scanning order 1: ', scanning_name_1)
print('Scanning order 2: ', scanning_name_2)
print('Scanning order 3: ', scanning_name_3)

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


################################################################################################################
# Starts calculations

if num_scan == 1:
    # calculating 1d scanning
    
    all_RGB = scan_1D_multiple(layer_scan_order['1'], t_scan_1, t_scan_step_1, structure, thickness, wavelengths, nk_dict, substrate)
    
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
    plt.xlabel(' / '.join(scanning_name_1) + ' \n Thickness (nm)', fontsize=8)
    titles = ''
    titles_1 = ''
    for i in range(len(structure)):
        titles += structure[i] 
        titles += '    '
        if i in layer_scan_order['0']:
            titles_1 += str(int(thickness[i]))
            titles_1 += 'nm  '
        else:
            titles_1 += ' ? nm  '
            
    plt.title(titles+'\n'+titles_1, fontsize=8)
    plt.savefig(file_name+'.jpg')

elif num_scan== 2:
    # calculating 2D scanning
    
    all_RGB = scan_2D_multiple(layer_scan_order['1'], layer_scan_order['2'], t_scan_1, t_scan_2, t_scan_step_1, t_scan_step_2, structure, thickness, wavelengths, nk_dict, substrate)
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
        
    plt.xlabel(' / '.join(scanning_name_1) + ' \n Thickness (nm)', fontsize=8)
    plt.ylabel(' / '.join(scanning_name_2) + ' \n Thickness (nm)', fontsize=8)
    titles = ''
    titles_1 = ''
    for i in range(len(structure)):
        titles += structure[i] 
        titles += '    '
        if i in layer_scan_order['0']:
            titles_1 += str(int(thickness[i]))
            titles_1 += 'nm  '
        else:
            titles_1 += ' ? nm  '
            
    plt.title(titles+'\n'+titles_1, fontsize=8)
    plt.savefig(file_name+'.jpg')

else:
    # calculating 3D scanning
    
    for scan_d in range(t_scan_3[0], t_scan_3[1], t_scan_step_3):
        

        for tt in layer_scan_order['3']:
            thickness[tt] = scan_d
        
        
        all_RGB = scan_2D_multiple(layer_scan_order['1'], layer_scan_order['2'], t_scan_1, t_scan_2, t_scan_step_1, t_scan_step_2, structure, thickness, wavelengths, nk_dict, substrate)
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
        plt.xlabel(' / '.join(scanning_name_1) + ' \n Thickness (nm)', fontsize=8)
        plt.ylabel(' / '.join(scanning_name_2) + ' \n Thickness (nm)', fontsize=8)
        titles = ''
        titles_1 = ''
        for i in range(len(structure)):
            titles += structure[i] 
            titles += '    '
            if i in layer_scan_order['0']:
                titles_1 += str(int(thickness[i]))
                titles_1 += 'nm  '
            else:
                titles_1 += ' ? nm  '
                
        plt.title(' / '.join(scanning_name_3) + '\nLayer 3 thickness = '+str(scan_d)+'nm\n'+titles+'\n'+titles_1, fontsize=8)
        
        plt.savefig(file_name+'_layer_3_'+str(scan_d)+'nm.jpg')
    



    
    


















