# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 23:30:10 2022

@author: mtg
"""

from numpy.core.fromnumeric import argsort
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution, sd_to_XYZ, XYZ_to_Lab
from scipy.optimize import differential_evolution
from colour.difference import delta_E_CIE2000, delta_E_CIE1976
import os
from os.path import exists


def delta_E_single(mixing_ratio):
    '''
    Args:
        mixing_ratio: the mixing weight for each sample
        r_data: spectrum for each sample
        target: target Lab value
    '''
    mixing_ratio = np.array([ii for ii in mixing_ratio])
    print(mixing_ratio, product_weights)
    mixing_ratio_ = mixing_ratio * product_weights / np.sum(mixing_ratio * product_weights)
    spectrum = np.sum(mixing_ratio_[:, None] * r_data, axis=0) / 100
    
    data = {wl:s for wl, s in zip(wavelengths, spectrum)}
    sd = SpectralDistribution(data)
    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
    lab = XYZ_to_Lab(XYZ /100, illuminant=np.array([0.31382, 0.33100]))

    deltaE = delta_E_loss(lab, target)

    total_weights = np.sum(mixing_ratio * product_weights)

    return np.max([0, deltaE - thresh]) - args.alpha * total_weights


def delta_E(mixing_ratio):
    '''
    Args:
        mixing_ratio: the mixing weight for each sample
        r_data: spectrum for each sample
        target: target Lab value
    '''
    mixing_ratio = np.array([ii for ii in mixing_ratio])
    #print(mixing_ratio, product_weights)
    mixing_ratio_ = mixing_ratio * product_weights / np.sum(mixing_ratio * product_weights)
    spectrum = np.sum(mixing_ratio_[:, None] * r_data, axis=0) / 100
    
    data = {wl:s for wl, s in zip(wavelengths, spectrum)}
    sd = SpectralDistribution(data)
    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
    lab = XYZ_to_Lab(XYZ /100, illuminant=np.array([0.31382, 0.33100]))

    deltaE = delta_E_loss(lab, target)

    total_weights = np.sum(mixing_ratio * product_weights)

    return np.max([0, deltaE - thresh]) - args.alpha * total_weights