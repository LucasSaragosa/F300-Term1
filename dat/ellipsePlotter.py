# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:38:46 2023


@author: BES (b.sherlock@exeter.ac.uk)

Script controls a ELL14 (https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL14) connected to the PC via USB cable -> Interface board -> ELLB bus distribution board 
The  ELL14 is at address '1' (note this is not the COM port) and all serial commands to this ELL are prefixed with a '0' e.g. '0in''


"""
import numpy as np
import serial
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from py_pol.utils import degrees
import os
scriptDir = os.getcwd()
# serialString = ""  # declare a string variable


# ARDser = serial.Serial(         # Open a serial connection to the Arduino
#     port='COM4',                # You may need to change the COM port depending on your computer
#     baudrate=9600
# )
# ARDser.reset_input_buffer()
# ARDser.flushInput()
# ARDser.flushOutput()



# powers = []
# jogStepDeg = 15
# polariserAngles = np.arange(0,180,jogStepDeg)

# for idx in tqdm(polariserAngles):    # In each iteration,  ELL14 takes a regular sized jog step
    
#     ARDser.write('pol'.encode())     #  Serial communication to Arduino 
#     pol = ARDser.readline().decode('ascii')     #   Serial communications back from the Arduino 
#     print('\n Power meter = ' + str(pol.strip()) + ' V')

#     powers.append(float(pol.strip()))   # Powers is a list that will contain all the different voltages recorded by the Arduino.
      
       
# # ELLser.close()
# ARDser.close()

import pandas as pd
folderName = scriptDir + '\\readings.csv'
data = pd.read_csv(folderName) 
#%%
plt.close('all')

# polariserAngles = np.arange(0,180,jogStepDeg)
polariserAngles = np.asarray(data.Degrees)*2
powers = np.asarray(data.Voltages)


fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(121)
plt.plot(polariserAngles,powers,'ro')
plt.xlabel('Polariser Angle (deg)')

from scipy.optimize import curve_fit
def model_f(theta,p1,p2,p3):        # This is the fitting routine used to fit the voltages from the Arduino. 
    
      return (p1*np.cos(theta*degrees-p3))**2 + (p2*np.sin(theta*degrees-p3))**2

popt, pcov = curve_fit(model_f, polariserAngles, powers, bounds = ([-5,-5,0],[5,5,np.pi]))

Emax, Emin, alpha = popt              # Emax = semi major axis of polarisation ellipse. Emin = semi minor axis of polarisation ellipse. Alpha = Angle of ellipse with respect to x axis 
fittingAngles = np.arange(0,np.max(polariserAngles),1)

plt.plot(fittingAngles,model_f(fittingAngles, Emax, Emin, alpha),'--b')    # This is to check the fitting routine worked, this isn't a plot of the ellipse itself

print('\n Ellipse semi major axis angle = ' + str(np.round(alpha*180/np.pi)) + ' degrees \n')


from matplotlib import patches
# fig = plt.figure()
ax = fig.add_subplot(122, aspect='auto')


e1 = patches.Ellipse((0, 0), Emax/3, Emin/3,
                  angle = alpha*180/np.pi, linewidth=2, fill = False, zorder=1)   # This is where we plot a "picture of the ellipse"

ax.add_patch(e1)
ax.set_xlim([-0.5,0.5])
ax.set_ylim([-0.5,0.5])
# ax.axis('square')
ax.axis('off')
plt.title('Polarisation ellipse')

print('Fitted Ex = ' + str(np.round(Emax,2)))
print('Fitted Ey = ' + str(np.round(Emin,2)))

#%% Save the data so thaqt fitting routines can be optimised offline (also later will need to save these data for reference)

# np.save('polariserAngles', polariserAngles)
# np.save('powers',powers)
