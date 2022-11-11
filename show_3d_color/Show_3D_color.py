
from numpy.core.fromnumeric import argsort
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution, sd_to_XYZ, XYZ_to_Lab,XYZ_to_RGB
from scipy.optimize import differential_evolution
from colour.difference import delta_E_CIE2000, delta_E_CIE1976
import os
from os.path import exists


cmfs = MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
illuminant = SDS_ILLUMINANTS['D65']
wavelengths = np.arange(400, 701, 10)


csv_name = 'WB1_side'  # change the name, 
if not exists(csv_name+'.csv'):
    print('The specificed file does not exist, please check your input. Make sure it is a csv file, and inside the same folder with this python file')
    raise NotADirectoryError
    
filename = csv_name

print('Loading file: '+filename+'.csv')

def calcualte_lab(spectrum):

    data = {wl:s for wl, s in zip(wavelengths, spectrum)}
    sd = SpectralDistribution(data)
    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
    lab = XYZ_to_Lab(XYZ /100, illuminant=np.array([0.31382, 0.33100]))
    return lab

def create_circle(radius):
    angles = np.linspace(0, 2.1*np.pi, 100)
    a = 50*np.ones([len(angles), 3])
    a[:, 1] = radius * np.cos(angles)
    a[:, 2] = radius * np.sin(angles)
    
    return a

data = pd.read_csv(csv_name+'.csv')
r_data = np.zeros((len(data), 31))
lab_data = np.zeros((len(data), 3))

for i in range(len(data)):
    for j, wl in enumerate(['{} nm'.format(int(wl)) for wl in np.arange(400, 701, 10)]):
        r_data[i, j] = data[wl].iloc[i]
    lab_data[i, :] = calcualte_lab(r_data[i, :]/100)



fig = plt.figure(dpi=150)
ax = plt.axes(projection='3d')
ax.set_axis_off()
ax.scatter3D(lab_data[:, 1], lab_data[:, 2], lab_data[:, 0], cmap='Greens') #x:a, y: b, z: L
ax.scatter3D(0, 0, 50, color='red', s=10)

ax.plot([-128, 128], [0, 0], [50, 50], color='black', linewidth=0.5)
ax.plot([0, 0], [-128, 128], [50, 50], color='black', linewidth=0.5)
ax.plot([0, 0], [0, 0], [0, 100], color='black', linewidth=0.5)

ax.text(0, 0, 0, "L\nBlack", color='black')
ax.text(0, 0, 105, "L\nWhite", color='black', alpha=0.3)
ax.text(-128-10, 0, 50, "-a\nGreen", color='green')
ax.text(128+10, 0, 50, "+a\nRed", color='red')
ax.text(0, -128-10, 50, "-b\nBlue", color='blue')
ax.text(0, 128+10, 50, "+b\nYellow", color='brown')

for i in range(5):
    a = create_circle(128/5*(i+1))
    ax.plot(a[:, 1], a[:, 2], a[:, 0], color='black', linewidth=0.5)

ax.legend()
ax.set_xlim([-128, 128])
ax.set_ylim([-128, 128])
ax.set_zlim([0, 100])
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('L')


