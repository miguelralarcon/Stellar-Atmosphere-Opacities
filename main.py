#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:16:06 2020

@author: mralarcon
"""


# Import packages
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path
import sys

# Define physical constants and conversion factors
c = 29979245800           # speed of light (cm/s)
K = 1.3806e-16            # boltzmann constant (erg/K)
R = 1.0968e-3             # rydberg constant (A^-1)
h = 6.626070150e-27       # plank constant (erg s)
eV_to_erg = 1.60218e-12   # 1 electronvolt (eV) in ergios (erg)


# Import atomic constants from aditional file
with open('atomicdata.txt', 'r') as f:
  flines = f.readlines()      # extract lines from file

# Extract information of ionization and excitation states
ion_index = flines.index('# Ionization states\n')
exc_index = flines.index('# Excitation states\n')
ion_data = [a.split() for a in flines[ion_index+2:exc_index-1]]
exc_data = [a.split() for a in flines[exc_index+2:]]

# Create dictionary structure
atom = {}
for i in ion_data:
  atom[i[0]] =  {'I': float(i[1])}
  g, X = [], []
  for j in [e for e in exc_data if e[0] == i[0]]:
    atom[i[0]].update({float(j[1]): {'X':float(j[2]), 'g':float(j[3])}})
    X.append(float(j[2]))
    g.append(float(j[3]))
  atom[i[0]].update({'X': np.array(X), 'g': np.array(g)})
  
  
#############################################################
#### ------ READING THE STELLAR ATMOSPHERE MODELS ------ ####
#############################################################
  
  
print('\nReading the stellar atmosphere models...')
headers, colnames, models = [], [], []

# List atmosphere files in folder
filelist = [f for f in listdir() if f[-4:]=='.dat']
filelist.sort()
for modfile in filelist:
  modln = []
  with open(modfile) as fl:
    filelines = fl.readlines()      # Extract lines from file

  # Read header
  teff = float(filelines[0].split()[0])
  flux = float(filelines[1].split()[0])  
  gravs = float(filelines[2].split()[0])
  abunH = float(filelines[3].split()[6][:-1])
  abunHe = float(filelines[3].split()[8])
  nd = int(filelines[4].split()[0])
  headers.append(np.array([teff, flux, gravs, abunH, abunHe]))

  # Read depth points
  colnames.append(filelines[6].split()[1:])
  for dl in filelines[7:len(filelines)]:
    modln.append(np.array(dl.split()[1:]).astype(np.float))

  # Show basic information
  print('\nFile name: %s' % modfile)
  print('Effective temperature: %i K' % teff)
  print('Total flux: %.4E erg/cm2/s' % flux)
  print('Number of depth points: %i' % np.array(modln).shape[0])
  print('Number of columns: %i' % np.array(modln).shape[1])

  # Check that all the depth points have been read
  if len(modln) != nd:
    print('Only %i/%i depth points read: Check the file %s!' %
          (len(modln), nd, modfile))
  else:
    models.append(np.array(modln))

print('\nDone!')
  
  
###################################################
#### ------ DEFINING PLOTTING FUNCTIONS ------ ####
###################################################

  
def plotter(ax, x, y, xlabel, ylabel, color, label, fontsize, tit = None,
            mk='None', lstyle='None', aph=1, bb=None,ncol=1,lw=3):
  # Format plt.axes for plotting the light curve
  ax.plot(x,y,color=color,label=label,marker=mk,linestyle=lstyle,alpha=aph,linewidth=lw)
  if label != None: ax.legend(loc=0,fontsize=fontsize-4,frameon=False,bbox_to_anchor = bb, ncol=ncol)
  if tit != None: ax.set_title(tit,fontsize=fontsize)
  if xlabel != None: ax.set_xlabel(xlabel,fontsize=fontsize)
  if ylabel != None: ax.set_ylabel(ylabel,fontsize=fontsize)

  ax.minorticks_on()
  ax.tick_params(axis='both',direction='inout',which='minor',
                 length=3,width=.5,labelsize=fontsize)
  ax.tick_params(axis='both',direction='inout',which='major',
                 length=8,width=1,labelsize=fontsize)
  ax.yaxis.set_ticks_position('both')
  ax.xaxis.set_ticks_position('both')

def plotter2(ax, x, y, xlabel, ylabel, color, label, fontsize, tit = None,
            mk='None', lstyle='None', aph=1, bb=None,ncol=1,lw=2):
  # Format plt.axes for plotting the light curve
  ax.plot(x,y,color=color,label=label,marker=mk,linestyle=lstyle,alpha=aph,linewidth=lw)
  if label != None: ax.legend(loc=0,fontsize=fontsize,frameon=False,bbox_to_anchor = bb, ncol=ncol)
  if tit != None: ax.set_title(tit,fontsize=fontsize)
  if xlabel != None: ax.set_xlabel(xlabel,fontsize=fontsize)
  if ylabel != None: ax.set_ylabel(ylabel,fontsize=fontsize)

  ax.minorticks_on()
  ax.tick_params(axis='both',direction='inout',which='minor',
                 length=3,width=.5,labelsize=fontsize)
  ax.tick_params(axis='both',direction='inout',which='major',
                 length=8,width=1,labelsize=fontsize)
  ax.yaxis.set_ticks_position('both')
  ax.xaxis.set_ticks_position('both')
  
  
  
  
#############################################
#### ------ OBTAINING POPULATIONS ------ ####
#############################################

from populations import *

print('\nObtaining populations...')

# Populations for each atmosphere depth point and model
atm = {}
for i,mod in enumerate(models):
  Narr, Nearr, narr = [], [], []
  for depth in mod:
    N, Ne, n = populations(atom, depth, abunH, abunHe)

    Narr.append(N), Nearr.append(Ne), narr.append(np.concatenate(n).ravel())
  Narr = np.array(Narr); Nearr = np.array(Nearr); narr = np.array(narr)
  
  # Update the dictionary
  atm[filelist[i]] = {'lgtaur': mod[:,0], 'lgtau5': mod[:,1],
                           'T': mod[:,2], 'Pe': mod[:,3], 'Ne': Nearr}
  for k,ke in enumerate(ion_data):
    atm[filelist[i]].update({ke[0]: {'N': Narr[:,k]}})
    for j,je in enumerate(exc_data):
      if je[0] == ke[0]:
        atm[filelist[i]][ke[0]].update({float(je[1]): {'N': narr[:,j]}})


## --- PRINTING POPULATIONS --- ##
        
# Print the populations
# (This code is a little bit messy and should be ignored)
ftsize = 20
xlabel = r'log $\tau_R$'
ylabel = 'log N'
lstyle = ['dashed','dotted','dashdot',(0, (3, 5, 1, 5)), (0, (3, 3, 1, 3))]
colormap = plt.cm.get_cmap('Spectral', 256)
color = colormap(np.linspace(0, 1, 7))
Nleg = None; nleg = None
fig = plt.figure(1, figsize=(16,5))
plt.clf()
for m,mod in enumerate(atm):
  tau = atm[mod]['lgtaur']
  ax = fig.add_subplot(1,2,m+1)
  i = 0
  for ion in atm[mod]:
    try: N = atm[mod][ion]['N']
    except: continue
    i += 1
    if i ==4:
      Ne = atm[mod]['Ne']
      if m==0: Neleg = None
      else: Neleg = 'Ne'
    if i < 4: co = color[i-1]
    else: co = color[3-i]
    if m == 0: Nleg = None
    if m == 1: Nleg = 'N (%s)' % ion; ylabel = None
    plotter(ax, tau, np.log10(N), xlabel, ylabel, co, Nleg, ftsize, lstyle='solid')
    ie = 0
    for e,exc in enumerate(atm[mod][ion]):
      if type(exc) != float: continue
      if m == 0: nleg = None
      else: nleg = 'n = %i' % int(exc)
      ie += 1
      n = atm[mod][ion][exc]['N']
      plotter(ax, tau, np.log10(n), xlabel, ylabel, co, nleg, ftsize,
              lstyle=lstyle[ie-1],bb=(1.05,1.03),ncol = 2)
  title = r'T$_{eff}$ = ' + mod[1:-4] + ' K'
  ax.text(0.5, 0.08, title, transform=ax.transAxes, fontsize=ftsize)
  ax.set_ylim(-80,25)
  
plt.tight_layout()
#plt.savefig(path+'populations.jpg',dpi=1000)
plt.show()


print('Done!')
  



###########################################
#### ------ OBTAINING OPACITIES ------ ####
###########################################

from opacities import *

print('\nObtaining opacities...')

# Add maximum wavelengths to dictionary
for ion in atom:
  for exc in atom[ion]:
    try: float(exc)
    except: continue
    if atom[ion]['I'] == 0: continue      # ignore the las ionization states
    wlA = c*h*1e8/((atom[ion]['I'] - atom[ion][exc]['X'])*eV_to_erg) # A
    print
    atom[ion][exc].update({'wlmaxA': wlA})
    
    
# Wavelength array
wl = np.linspace(100, 1e5, 10000)
wl2 = np.array([(1 - 1e-4)*atom['HeI'][1]['wlmaxA'], (1 + 1e-4)*atom['HeI'][1]['wlmaxA'], (1 - 1e-4)*1/R, (1 + 1e-4)*1/R, (1 - 1e-4)*4/R,
                (1 + 1e-4)*4/R])
wl3 = np.array([3640]) 

# Opacity dictionaries
opacity_dict_gen = opacity_func(wl, atm, atom, filelist)
opacity_dict_tab = opacity_func(wl2, atm, atom, filelist)
opacity_dict_3640 = opacity_func(wl3, atm, atom, filelist)


## --- PRINTING TOTAL OPACITY vs LAMBDA --- ##

# Index for which Rosseland optical depth = 1
index_taur1_1 = np.where(opacity_dict_gen['t5000.dat']['lgtaur'] == 0)
index_taur1_2 = np.where(opacity_dict_gen['t8000.dat']['lgtaur'] == 0)

# Total opacity
tot_5000 = opacity_dict_gen['t5000.dat']['tot'][index_taur1_1[0][0]]
tot_8000 = opacity_dict_gen['t8000.dat']['tot'][index_taur1_1[0][0]]

# Print
ftsize = 14
xlabel = r'log $\lambda(\AA)$'
ylabel = r'log $\kappa(cm^{-1})$'
colormap = plt.cm.get_cmap('Spectral', 256)
color = colormap(np.linspace(0, 1, 7))
fig = plt.figure(2, figsize=(6,5.5))
plt.clf()
ax = fig.add_subplot(1,1,1)
plotter2(ax, np.log10(wl), np.log10(tot_5000), xlabel, ylabel, 'r',
        r'5000 K', ftsize, lstyle='solid')
plotter2(ax, np.log10(wl), np.log10(tot_8000), xlabel, ylabel, 'b',
        r'8000 K', ftsize, lstyle='solid')
ax.set_xlim(np.log10(500),np.log10(20000))
plt.tight_layout()
#plt.savefig(path+'total_opacity_lambda.jpg',dpi=1000) 
plt.show()


## --- PRINTING OPACITY vs LAMBDA FOR EACH LEVEL --- ##

# Index for which Rosseland optical depth = 1
mod = 't5000.dat'
index_taur1_1 = np.where(opacity_dict_gen[mod]['lgtaur'] == 0)
H_ff_5000 = opacity_dict_gen[mod]['H-']['ff'][index_taur1_1[0][0]]
HI_ff_5000 = opacity_dict_gen[mod]['HII']['ff'][index_taur1_1[0][0]]
HeI_ff_5000 = opacity_dict_gen[mod]['HeII']['ff'][index_taur1_1[0][0]]
HeII_ff_5000 = opacity_dict_gen[mod]['HeIII']['ff'][index_taur1_1[0][0]]
H_bf_5000 = opacity_dict_gen[mod]['H-'][1.0]['bf'][index_taur1_1[0][0]]
HI_1_bf_5000 = opacity_dict_gen[mod]['HI'][1.0]['bf'][index_taur1_1[0][0]]
HI_2_bf_5000 = opacity_dict_gen[mod]['HI'][2.0]['bf'][index_taur1_1[0][0]]
HI_3_bf_5000 = opacity_dict_gen[mod]['HI'][3.0]['bf'][index_taur1_1[0][0]]
HI_4_bf_5000 = opacity_dict_gen[mod]['HI'][4.0]['bf'][index_taur1_1[0][0]]
HeI_1_bf_5000 = opacity_dict_gen[mod]['HeI'][1.0]['bf'][index_taur1_1[0][0]]
HeI_2_bf_5000 = opacity_dict_gen[mod]['HeI'][2.0]['bf'][index_taur1_1[0][0]]
HeI_3_bf_5000 = opacity_dict_gen[mod]['HeI'][3.0]['bf'][index_taur1_1[0][0]]
HeI_4_bf_5000 = opacity_dict_gen[mod]['HeI'][4.0]['bf'][index_taur1_1[0][0]]
HeI_5_bf_5000 = opacity_dict_gen[mod]['HeI'][5.0]['bf'][index_taur1_1[0][0]]
HeII_1_bf_5000 = opacity_dict_gen[mod]['HeII'][1.0]['bf'][index_taur1_1[0][0]]
HeII_2_bf_5000 = opacity_dict_gen[mod]['HeII'][2.0]['bf'][index_taur1_1[0][0]]
Thom_5000 = opacity_dict_gen[mod]['Thom'][index_taur1_1[0][0]]*np.ones_like(HeII_2_bf_5000)

opacity_5000 = [H_ff_5000, HI_ff_5000, HeI_ff_5000, HeII_ff_5000, H_bf_5000, HI_1_bf_5000,
                HI_2_bf_5000, HI_3_bf_5000, HI_4_bf_5000, HeI_1_bf_5000, HeI_2_bf_5000,
                HeI_3_bf_5000, HeI_4_bf_5000, HeI_5_bf_5000, HeII_1_bf_5000, HeII_2_bf_5000,
                Thom_5000, tot_5000]

labels_mmgm = [r'H$^{ff}$', r'HI$^{ff}$', r'HeI$^{ff}$', r'HeII$^{ff}$', r'H$^{bf}$', r'HI$_1^{bf}$', r'HI$_2^{bf}$',
               r'HI$_3^{bf}$', r'HI$_4^{bf}$', r'HeI$_1^{bf}$', r'HeI$_2^{bf}$', r'HeI$_3^{bf}$', r'HeI$_4^{bf}$',
               r'HeI$_5^{bf}$', r'HeII$_1^{bf}$', r'HeII$_2^{bf}$', 'Thomson', 'Total']

Thom_8000 = opacity_dict_gen['t8000.dat']['Thom'][index_taur1_2[0][0]]
H_ff_8000 = opacity_dict_gen['t8000.dat']['H-']['ff'][index_taur1_2[0][0]]
HI_ff_8000 = opacity_dict_gen['t8000.dat']['HII']['ff'][index_taur1_2[0][0]]
HeI_ff_8000 = opacity_dict_gen['t8000.dat']['HeII']['ff'][index_taur1_2[0][0]]
HeII_ff_8000 = opacity_dict_gen['t8000.dat']['HeIII']['ff'][index_taur1_2[0][0]]
H_bf_8000 = opacity_dict_gen['t8000.dat']['H-'][1.0]['bf'][index_taur1_2[0][0]]
HI_1_bf_8000 = opacity_dict_gen['t8000.dat']['HI'][1.0]['bf'][index_taur1_2[0][0]]
HI_2_bf_8000 = opacity_dict_gen['t8000.dat']['HI'][2.0]['bf'][index_taur1_2[0][0]]
HI_3_bf_8000 = opacity_dict_gen['t8000.dat']['HI'][3.0]['bf'][index_taur1_2[0][0]]
HI_4_bf_8000 = opacity_dict_gen['t8000.dat']['HI'][4.0]['bf'][index_taur1_2[0][0]]
HeI_1_bf_8000 = opacity_dict_gen['t8000.dat']['HeI'][1.0]['bf'][index_taur1_2[0][0]]
HeI_2_bf_8000 = opacity_dict_gen['t8000.dat']['HeI'][2.0]['bf'][index_taur1_2[0][0]]
HeI_3_bf_8000 = opacity_dict_gen['t8000.dat']['HeI'][3.0]['bf'][index_taur1_2[0][0]]
HeI_4_bf_8000 = opacity_dict_gen['t8000.dat']['HeI'][4.0]['bf'][index_taur1_2[0][0]]
HeI_5_bf_8000 = opacity_dict_gen['t8000.dat']['HeI'][5.0]['bf'][index_taur1_2[0][0]]
HeII_1_bf_8000 = opacity_dict_gen['t8000.dat']['HeII'][1.0]['bf'][index_taur1_2[0][0]]
HeII_2_bf_8000 = opacity_dict_gen['t8000.dat']['HeII'][2.0]['bf'][index_taur1_2[0][0]]
Thom_8000 = opacity_dict_gen['t8000.dat']['Thom'][index_taur1_2[0][0]]*np.ones_like(HeII_2_bf_8000)

opacity_8000 = [H_ff_8000, HI_ff_8000, HeI_ff_8000, HeII_ff_8000, H_bf_8000, HI_1_bf_8000,
                HI_2_bf_8000, HI_3_bf_8000, HI_4_bf_8000, HeI_1_bf_8000, HeI_2_bf_8000,
                HeI_3_bf_8000, HeI_4_bf_8000, HeI_5_bf_8000, HeII_1_bf_8000, HeII_2_bf_8000,
                Thom_8000, tot_8000]


colormap = plt.cm.get_cmap('Spectral', 256)
color = colormap(np.linspace(0, 1, 25))
fig = plt.figure(3, figsize=(18,6))
plt.clf()
ftsize = 20

ax1 = fig.add_subplot(1,2,1)
x = np.log10(wl)
i=0
for j,lab in enumerate(labels_mmgm):
  opacity_5000[j][np.where(opacity_5000[j] <= 0)] = 1e-200
  y = np.log10(opacity_5000[j])
  lstyle='solid'; co = color[j]
  if j in [2,3,9,10,11,12,13,14,15]:
    i += 1
    co = color[25-i]
  if j > 3 and j not in [16,17]: lstyle='dashed'
  if j == 17: co = 'k'
  if j == 16: co = 'gray'
  plotter(ax1, x, y, xlabel, ylabel, co, None, ftsize, lstyle=lstyle)
  title = r'T$_{eff}$ = 5000 K'
ax1.text(0.5, 0.92, title, transform=ax1.transAxes, fontsize=ftsize)
ax1.set_xlim(2,4.5)
ax1.set_ylim(-75,5)

ax2 = fig.add_subplot(1,2,2)
i = 0
for j,lab in enumerate(labels_mmgm):
  opacity_8000[j][np.where(opacity_8000[j] <= 0)] = 1e-200
  y = np.log10(opacity_8000[j])
  lstyle='solid'; co = color[j]
  if j in [2,3,9,10,11,12,13,14,15]:
    i += 1
    co = color[25-i]
  if j > 3 and j not in [16,17]: lstyle='dashed'
  if j == 17: co = 'k'
  if j == 16: co = 'gray'
  plotter(ax2, x, y, xlabel, None, co, lab, ftsize, lstyle=lstyle,bb=(1.05,0.7,0.5,0.1),ncol = 2)
  title = r'T$_{eff}$ = 8000 K'
ax2.text(0.5, 0.92, title, transform=ax2.transAxes, fontsize=ftsize)
ax2.set_xlim(2,4.5)
ax2.set_ylim(-75,5)
plt.tight_layout()
#plt.savefig(path+'opacity_lambda.jpg',dpi=1000)
plt.show()



## --- PRINTING OPACITY vs TEMPERATURE FOR EACH LEVEL --- ##

y_3640_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['tot']])
y_Thom_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['Thom']])
y_H_ff_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['H-']['ff']])
y_HI_ff_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HII']['ff']])
y_HeI_ff_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeII']['ff']])
y_HeII_ff_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeIII']['ff']])
y_H_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['H-'][1.0]['bf']])
y_HI_1_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HI'][1.0]['bf']])
y_HI_2_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HI'][2.0]['bf']])
y_HI_3_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HI'][3.0]['bf']])
y_HI_4_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HI'][4.0]['bf']])
y_HeI_1_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeI'][1.0]['bf']])
y_HeI_2_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeI'][2.0]['bf']])
y_HeI_3_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeI'][3.0]['bf']])
y_HeI_4_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeI'][4.0]['bf']])
y_HeI_5_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeI'][5.0]['bf']])
y_HeII_1_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeII'][1.0]['bf']])
y_HeII_2_bf_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['HeII'][2.0]['bf']])

ovsT_5000 = [y_H_ff_5000, y_HI_ff_5000, y_HeI_ff_5000, y_HeII_ff_5000, y_H_bf_5000, y_HI_1_bf_5000,
             y_HI_2_bf_5000, y_HI_3_bf_5000, y_HI_4_bf_5000, y_HeI_1_bf_5000, y_HeI_2_bf_5000,
             y_HeI_3_bf_5000, y_HeI_4_bf_5000, y_HeI_5_bf_5000, y_HeII_1_bf_5000, y_HeII_2_bf_5000,
             y_Thom_5000, y_3640_5000]

T_5000 = opacity_dict_3640 ['t5000.dat']['T']

y_3640_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['tot']])
y_Thom_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['Thom']])
y_H_ff_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['H-']['ff']])
y_HI_ff_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HII']['ff']])
y_HeI_ff_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeII']['ff']])
y_HeII_ff_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeIII']['ff']])
y_H_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['H-'][1.0]['bf']])
y_HI_1_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HI'][1.0]['bf']])
y_HI_2_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HI'][2.0]['bf']])
y_HI_3_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HI'][3.0]['bf']])
y_HI_4_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HI'][4.0]['bf']])
y_HeI_1_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeI'][1.0]['bf']])
y_HeI_2_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeI'][2.0]['bf']])
y_HeI_3_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeI'][3.0]['bf']])
y_HeI_4_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeI'][4.0]['bf']])
y_HeI_5_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeI'][5.0]['bf']])
y_HeII_1_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeII'][1.0]['bf']])
y_HeII_2_bf_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['HeII'][2.0]['bf']])


ovsT_8000 = [y_H_ff_8000, y_HI_ff_8000, y_HeI_ff_8000, y_HeII_ff_8000, y_H_bf_8000, y_HI_1_bf_8000,
             y_HI_2_bf_8000, y_HI_3_bf_8000, y_HI_4_bf_8000, y_HeI_1_bf_8000, y_HeI_2_bf_8000,
             y_HeI_3_bf_8000, y_HeI_4_bf_8000, y_HeI_5_bf_8000, y_HeII_1_bf_8000, y_HeII_2_bf_8000,
             y_Thom_8000, y_3640_8000]

T_8000 = opacity_dict_3640 ['t8000.dat']['T']


colormap = plt.cm.get_cmap('Spectral', 256)
color = colormap(np.linspace(0, 1, 25))
fig = plt.figure(4, figsize=(18,6))
plt.clf()
ftsize = 20
xlabel = r'log T(K)'

ax1 = fig.add_subplot(1,2,1)
x = np.log10(T_5000)
i=0
for j,lab in enumerate(labels_mmgm):
  ovsT_5000[j][np.where(ovsT_5000[j] <= 0)] = 1e-200
  y = np.log10(ovsT_5000[j])
  lstyle='solid'; co = color[j]
  if j in [2,3,9,10,11,12,13,14,15]:
    i += 1
    co = color[25-i]
  if j > 3 and j not in [16,17]: lstyle='dashed'
  if j == 17: co = 'k'
  if j == 16: co = 'gray'
  plotter(ax1, x, y, xlabel, ylabel, co, None, ftsize, lstyle=lstyle)
title = r'T$_{eff}$ = 5000 K'
ax1.text(0.5, 0.92, title, transform=ax1.transAxes, fontsize=ftsize)
ax1.set_xlim(3.52,3.98)
ax1.set_ylim(-75,5)


ax2 = fig.add_subplot(1,2,2)
x = np.log10(T_8000)
i = 0
for j,lab in enumerate(labels_mmgm):
  ovsT_8000[j][np.where(ovsT_8000[j] <= 0)] = 1e-200
  y = np.log10(ovsT_8000[j])
  lstyle='solid'; co = color[j]
  if j in [2,3,9,10,11,12,13,14,15]:
    i += 1
    co = color[25-i]
  if j > 3 and j not in [16,17]: lstyle='dashed'
  if j == 17: co = 'k'
  if j == 16: co = 'gray'
  plotter(ax2, x, y, xlabel, None, co, lab, ftsize, lstyle=lstyle,bb=(1.05,0.7,0.5,0.1),ncol = 2)
title = r'T$_{eff}$ = 8000 K'
ax2.text(0.5, 0.92, title, transform=ax2.transAxes, fontsize=ftsize)
ax2.set_xlim(3.73,4.4)
ax2.set_ylim(-75,5)
plt.tight_layout()
#plt.savefig(path+'opacity_T.jpg',dpi=1000)


## --- PRINTING TOTAL OPACITY vs TEMPERATURE --- ##


y_3640_5000 = np.array([op for op in opacity_dict_3640['t5000.dat']['tot']])
y_3640_8000 = np.array([op for op in opacity_dict_3640['t8000.dat']['tot']])
T_5000 = opacity_dict_3640 ['t5000.dat']['T']
T_8000 = opacity_dict_3640 ['t8000.dat']['T']

ftsize = 14
xlabel = r'log T(K)'
ylabel = r'log $\kappa(cm^{-1})$'
fig = plt.figure(5, figsize=(6,5.5))
plt.clf()

ax = fig.add_subplot(1,1,1)
plotter2(ax, np.log10(T_5000), np.log10(y_3640_5000), xlabel, ylabel, 'r',
        r'5000 K', ftsize, lstyle='solid')
plotter2(ax, np.log10(T_8000), np.log10(y_3640_8000), xlabel, ylabel, 'b',
        r'8000 K', ftsize, lstyle='solid')
plt.tight_layout()
#plt.savefig(path+'total_opacity_T.jpg',dpi=1000)
plt.show()


print('Done!')