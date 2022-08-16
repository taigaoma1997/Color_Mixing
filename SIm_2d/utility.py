# -*- coding: utf-8 -*-
"""
Created on Mon Aug 5 21:21:46 2022

@author: mtg
"""

from tmm import coh_tmm, inc_tmm
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import pandas as pd
import colour
import os
from colour.plotting import plot_chromaticity_diagram_CIE1931, ColourSwatch, plot_multi_colour_swatches, plot_sds_in_chromaticity_diagram_CIE1931
from colour.plotting.models import plot_RGB_colourspaces_in_chromaticity_diagram
from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution



DATABASE = './data'
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']



def load_materials(all_mats, wavelengths):
    '''
    Load material nk and return corresponding interpolators.

    Return:
        nk_dict: dict, key -- material name, value: n, k in the 
        self.wavelength range
    '''
    nk_dict = {}

    for mat in all_mats:
        nk = pd.read_csv(os.path.join(DATABASE, mat + '.csv'))
        nk.dropna(inplace=True)

        wl = nk['wl'].to_numpy()
        index_n = nk['n'].to_numpy()
        index_k = nk['k'].to_numpy()

        n_fn = interp1d(
                wl, index_n,  bounds_error=False, fill_value='extrapolate', kind=3)
        k_fn = interp1d(
                wl, index_k,  bounds_error=False, fill_value='extrapolate', kind=1)
            
        nk_dict[mat] = n_fn(wavelengths) + 1j*k_fn(wavelengths)
    nk_dict['Air'] = np.ones_like(nk_dict['SiO2'])


    return nk_dict

    

def spectrum(materials, thickness, theta, pol, wavelengths, nk_dict,  substrate = 'Glass'):
    '''
    Input:
        materials: list
        thickness: list
        mixing_ratio: list
        theta: degree, the incidence angle

    Return:
        s: array, spectrum
    '''

    thickness = [np.inf] + thickness + [np.inf]
    
    degree = np.pi/180

    R, T, A, LAB, RGB = [], [], [], [], []
    for i, lambda_vac in enumerate(wavelengths * 1e3):

        nk_wave = [nk_dict[mat][i] for mat in materials]

        inc_list = ['i'] + ['c']*len(nk_wave) + ['i', 'i']
        # n_list = [1] + nk_wave + [nk_dict[substrate][i], 1]
        n_list = [1] + nk_wave + [nk_dict[substrate][i]]

        res = coh_tmm(pol, n_list, thickness, theta * degree, lambda_vac)
        # res = inc_tmm(pol, n_list, thickness, inc_list, theta * degree, lambda_vac)

        R.append(res['R'])
        T.append(res['T'])

    R, T = np.array(R), np.array(T)
    A = 1 - R - T


    All_result = {'R':R, 'T':T, 'A':A}
    return All_result



def get_color(R, wavelengths):
    # return the xyY, RGB, LAB from reflection
    data = dict(zip((1e3 * wavelengths).astype('int'), R))
    sd = SpectralDistribution(data)

    XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant)
    xyY = colour.XYZ_to_xyY(XYZ)
    Lab = colour.XYZ_to_Lab(XYZ / 100)
    RGB = colour.XYZ_to_sRGB(XYZ / 100)


    return Lab, RGB, xyY

def scan_1D(num_scan, t_scan_1, t_scan_step_1, structure, thickness, wavelengths, nk_dict, substrate):
    # scanning 1D thickness
    
    all_RGB = []
    all_xyY = []
    
    temp_RGB = []
    temp_xyY = []

    for j in range(t_scan_1[0], t_scan_1[1]+1, t_scan_step_1):
        thickness[num_scan[0]] = j
        spec_s = spectrum(structure, thickness, theta = 0, pol = 's', wavelengths = wavelengths, nk_dict = nk_dict,  substrate = substrate)
        spec_p = spectrum(structure, thickness, theta = 0, pol = 'p', wavelengths = wavelengths, nk_dict = nk_dict,  substrate = substrate)
        R = (spec_s['R'] + spec_p['R'])/2
        Lab, RGB, xyY = get_color(R, wavelengths)
        temp_RGB.append(list(RGB))
        temp_xyY.append(list(xyY))

    all_RGB.append(temp_RGB)
    all_xyY.append(temp_xyY)

    return all_RGB 




def scan_2D(num_scan, t_scan_1, t_scan_2, t_scan_step_1, t_scan_step_2, structure, thickness, wavelengths, nk_dict, substrate):
    # scanning 2D thickness
    
    all_RGB = []
    all_xyY = []

    for i in tqdm(range(t_scan_2[0], t_scan_2[1]+1, t_scan_step_2)):
        thickness[num_scan[1]] = i
        temp_RGB = []
        temp_xyY = []

        for j in range(t_scan_1[0], t_scan_1[1]+1, t_scan_step_1):
            thickness[num_scan[0]] = j
            spec_s = spectrum(structure, thickness, theta = 0, pol = 's', wavelengths = wavelengths, nk_dict = nk_dict,  substrate = substrate)
            spec_p = spectrum(structure, thickness, theta = 0, pol = 'p', wavelengths = wavelengths, nk_dict = nk_dict,  substrate = substrate)
            R = (spec_s['R'] + spec_p['R'])/2
            Lab, RGB, xyY = get_color(R, wavelengths)
            temp_RGB.append(list(RGB))
            temp_xyY.append(list(xyY))

        all_RGB.append(temp_RGB)
        all_xyY.append(temp_xyY)

    all_RGB = all_RGB[::-1]
    return all_RGB 



