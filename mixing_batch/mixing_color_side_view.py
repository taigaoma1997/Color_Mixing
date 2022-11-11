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


#############################################################################
# Please only change parameters inside this cell

alpha_weight = 0.0001   # change the alpha factor 
alpha_side = 1.0

Delta_E = 2.0
Delta_E_side = 2.0

target = [68.79, -1.14, 2.5]  #
target_side = [79.175, -7.48, 1.745]  #

csv_name = 'WB1_side'  # change the name, 
delta_E_type = 'CIE1976' # by default the delta_E_type is CIE1976
num_cpu = 4 # specify the number of CPU cores


#############################################################################

cmfs = MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
illuminant = SDS_ILLUMINANTS['D65']
wavelengths = np.arange(400, 701, 10)

#############################################################################
# check if the input arguments are correct

if not exists(csv_name+'.csv'):
    print('The specificed file does not exist, please check your input. Make sure it is a csv file, and inside the same folder with this python file')
    raise NotADirectoryError
    
filename = csv_name
if not exists(filename):
    os.mkdir(filename)

if delta_E_type == 'CIE2000':
    delta_E_loss = delta_E_CIE2000
elif delta_E_type == 'CIE1976':
    delta_E_loss = delta_E_CIE1976
else:
    print('The CIE difference only support delta_E_CIE2000 and delta_E_CIE1976, please check your input')
    raise NotImplementedError
    
#############################################################################
# The following part is used to load files 
# Do not change
# Make sure the file is in .csv format 

print('Loading file: '+filename+'.csv')
data = pd.read_csv(csv_name+'.csv')
r_data = np.zeros((len(data), 31))
r_side_data = np.zeros((len(data), 31))
weight = []
for i in range(len(data)):
    for j, wl in enumerate(['{} nm'.format(int(wl)) for wl in np.arange(400, 701, 10)]):
        r_data[i, j] = data[wl].iloc[i]
    for j, wl in enumerate(['{}nm_'.format(int(wl)) for wl in np.arange(400, 701, 10)]):
        r_side_data[i, j] = data[wl].iloc[i]
        
product_weights = data['Weight'].to_numpy()[1:]

r_data_target = r_data[0]
r_data = r_data[1:]

r_side_data_target = r_side_data[0]
r_side_data = r_side_data[1:]


#############################################################################

# target Lab value
target = np.array(target)
target_side = np.array(target_side)
print('Target LAB value: ', target)
print('Delta E target: ', Delta_E)
print('Target LAB value (side): ', target_side)
print('Delta E side target (side): ', Delta_E_side)

print('alpha weight: ', alpha_weight)
print('alpha side: ', alpha_side)
print('CIE loss type:', delta_E_type)
print('Starting optimization...................')

#############################################################################
# define a function to calculate the LABafter fining the ratios 
# Do not change 

def calcualte_lab(result, r_datas):
    mixing_ratio = result.x
    total_weights = np.sum(mixing_ratio * product_weights)
    mixing_ratio = mixing_ratio * product_weights / total_weights # normalize the mixing_ratio vector so that all dimensions sum up to 1
    spectrum = np.sum(mixing_ratio[:, None] * r_datas, axis=0) / 100

    data = {wl:s for wl, s in zip(wavelengths, spectrum)}
    sd = SpectralDistribution(data)
    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
    lab = XYZ_to_Lab(XYZ /100, illuminant=np.array([0.31382, 0.33100]))
    return lab

#############################################################################

def delta_E(mixing_ratio):
    '''
    Args:
        mixing_ratio: the mixing weight for each sample
        r_data: spectrum for each sample
        target: target Lab value
    '''

    mixing_ratio_ = mixing_ratio * product_weights / np.sum(mixing_ratio * product_weights)
    spectrum = np.sum(mixing_ratio_[:, None] * r_data, axis=0) / 100
    
    data = {wl:s for wl, s in zip(wavelengths, spectrum)}
    sd = SpectralDistribution(data)
    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
    lab = XYZ_to_Lab(XYZ /100, illuminant=np.array([0.31382, 0.33100]))

    deltaE = delta_E_loss(lab, target)
    
    spectrum_side = np.sum(mixing_ratio_[:, None] * r_side_data, axis=0) / 100
    
    data_side = {wl:s for wl, s in zip(wavelengths, spectrum_side)}
    sd_side = SpectralDistribution(data_side)
    XYZ_side = sd_to_XYZ(sd_side, cmfs, illuminant)
    lab_side = XYZ_to_Lab(XYZ_side /100, illuminant=np.array([0.31382, 0.33100]))

    deltaE_side = delta_E_loss(lab_side, target_side)

    total_weights = np.sum(mixing_ratio * product_weights)

    return np.max([0, deltaE - Delta_E]) + alpha_side * np.max([0, deltaE_side - Delta_E_side]) - alpha_weight * total_weights



if __name__ == '__main__':
        
    def optimize_mixing():

        bounds = [(0, 1)] * len(r_data)
        result = differential_evolution(delta_E, bounds, disp=True, workers=num_cpu, maxiter=2, popsize=16, tol=0.001)
        
        return result


    result = optimize_mixing()
    lab = calcualte_lab(result, r_data)
    lab_side = calcualte_lab(result, r_side_data)
    
    print('Saving figures')
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    ax[0].plot(wavelengths, (result.x[:, None] / result.x.sum() * r_data).sum(axis=0))
    ax[0].plot(wavelengths, r_data_target)
    
    ax[0].legend(['Mixed Lab = [ {:.2f}, {:.2f}, {:.2f}]'.format(lab[0], lab[1], lab[2]), f'Target, Lab = {target}'])
    ax[0].set_xlabel('Wavelength (nm)')
    ax[0].set_ylabel('Reflection (%)')
    ax[0].set_ylim([0, 100])
    ax[0].set_title('Normal view \nTotal weights {:.2f}, DeltaE={:.2f}'.format(np.sum(result.x * product_weights) / np.sum(product_weights), delta_E_loss(lab, target)))
    
    ax[1].plot(wavelengths, (result.x[:, None] / result.x.sum() * r_side_data).sum(axis=0))
    ax[1].plot(wavelengths, r_side_data_target)
    
    ax[1].legend(['Mixed Lab = [ {:.2f}, {:.2f}, {:.2f}]'.format(lab_side[0], lab_side[1], lab_side[2]), f'Target, Lab = {target}'])
    ax[1].set_xlabel('Wavelength (nm)')
    ax[1].set_ylabel('Reflection (%)')
    ax[1].set_ylim([0, 100])
    ax[1].set_title('Side view \nTotal weights {:.2f}, DeltaE_side={:.2f}'.format(np.sum(result.x * product_weights) / np.sum(product_weights), delta_E_loss(lab_side, target)))
    
    ax[2].stem(result.x)
    ax[2].set_xlabel('Batch ID')
    ax[2].set_ylabel('Mixing ratio (used / all)')
    plt.savefig('./{}/{}_DeltaE{}_DeltaE_side_{}_alpha_weight_{}_alpha_side_{}.png'.format(filename,  delta_E_type, Delta_E, Delta_E_side, alpha_weight, alpha_side))
    
    
    data = pd.read_csv(csv_name+'.csv')
    x = result.x
    product_weights_ = np.insert(product_weights, 0, np.inf)
    x = np.insert(x, 0, 1)
    data['mixing weight'] = np.round(x * product_weights_, 1)
    data.sort_values('mixing weight', ascending=False).to_csv('./{}/{}_mixing_results_{}_DeltaE{}_DeltaE_side_{}_alpha_weight_{}_alpha_side_{}.csv'.format(filename, filename, delta_E_type, Delta_E, Delta_E_side, alpha_weight, alpha_side), index=None)
    print('Done')