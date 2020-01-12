#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esto código permite obtener las opacidades de los niveles de excitación de
los diferentes estados de ionización de los elementos presentes en un modelo
de atmósfera dado.

@author: mmgomezmiguez & mralarcon
@date: Jan. 2020
"""

# Import packages
import numpy as np

# Define physical constants and conversion factors
c = 29979245800           # speed of light (cm/s)
K = 1.3806e-16            # boltzmann constant (erg/K)
R = 1.0968e-3             # rydberg constant (A^-1)
h = 6.626070150e-27       # plank constant (erg s)
eV_to_erg = 1.60218e-12   # 1 electronvolt (eV) in ergios (erg)


##########################################
#### ------ DEFINING FUNCTIONS ------ ####
##########################################


## --- GAUNT COEFFICIENTS --- ##

def G_bf(wl, n):
# Function to measure Gaunt bound-free coefficient for hydrogen type atoms
# Input:
#    wl: array of wavelengths in A
#     n: principal quantum number
# Output:
#  g_bf: Gaunt bound-free coefficient
# mmgomezmiguez & mralarcon, Jan. 2020
  g_bf = 1 - 0.3456/(wl*R)**(1/3)*(wl*R/n**2 - 0.5)
  return g_bf

def G_ff(wl, T):
# Function to measure Gaunt free-free coefficient for hydrogen type atoms
# Input:
#    wl: array of wavelength in A
#     n: principal quantum number
# Output:
#  g_ff: Gaunt free-free coefficient
# mmgomezmiguez & mralarcon, Jan. 2020
  g_ff = 1 + 0.3456/(wl*R)**(1/3)*(wl*K*T/(h*c*1e8) + 0.5)
  return g_ff



## --- CROSS SECTIONS --- ##
  
def Sigma_bf(wl, wl_max, Z, n, g_bf):
# Function to measure bound-free effective section for hydrogen type atoms
# Input:
#     wl: array of wavelength in A
# wl_max: Balmer's jump wavelenth
#      Z: number of protons of the atom
#      n: principal quantum number
#   g_bf: Gaunt bound-free coefficient
# Output:
#  sigma_bf: bound-free effective section
# mmgomezmiguez & mralarcon, Jan. 2020
  sigma_bf = 2.815e29*Z**4*wl**3/(n**5*(c*1e8)**3)*g_bf
  #Balmer's limit
  sigma_bf[wl>=wl_max] = 0.

  return sigma_bf

def Sigma_ff(wl, T, Z, g_ff):
# Function to measure free-free effective section for hydrogen type atoms
# Input:
#    wl: array of wavelength in A
#     T: temperature in K
#     Z: number of protons of the atom
#  g_bf: Gaunt bound-free coefficient
# Output:
#  sigma_ff: free-free effective section
# mmgomezmiguez & mralarcon, Jan. 2020 
  sigma_ff = 3.7e8*Z**2*wl**3/(T**0.5*(c*1e8)**3)*g_ff

  return sigma_ff

def Sigma_HeI_n1(wl, wl_max):
# Function to measure HeI effective section coefficients for n=1
# Input:
#    wl: array of wavelength in A
# Output:
#  sigma_HeI: total effective section
# mmgomezmiguez & mralarcon, Jan. 2020
    sigma_HeI = 2.951209e14*wl**2/(c*1e8)**2
    sigma_HeI[wl>=wl_max] = 0.
    return sigma_HeI

def Sigma_bf_H_(wl, wl_max):
# Function to measure bound-free effective section for H- ion
# Input:
#     wl: array of wavelength in A
# wl_max: Balmer's jump wavelenth
# Output:
#  sigma_bf_H_: bound-free effective section
# mmgomezmiguez & mralarcon, Jan. 2020

  # Coefficients
  a0 = 1.99654;       a4 = 3.23992e-14;
  a1 = -1.18267e-5;   a5 = -1.39568e-18;
  a2 = 2.64243e-6;    a6 = 2.78701e-23;
  a3 = -4.40524e-10;

  a = np.array([a0, a1, a2, a3, a4, a5, a6])*1e-18
  b = np.arange(7)

  # Calculate sigma_bf_H_
  pot = wl**b.reshape(len(b),1)
  sigma_bf_H_ = np.einsum('i,ij', a, pot)
  sigma_bf_H_[wl>=wl_max] = 0.

  return sigma_bf_H_


## --- OPACITIES --- ##

def Opacity_ff_H_(wl, Pe, n, T):
# Function to measure free-free effective section for H- ion
# Input:
#    wl: array of wavelengths in A
#    Pe: electron pressure (erg/cm^3)
#     n: population of certain level
#     T: temperature in K
# Output:
#  sigma_ff_H_: free-free opacity
# mmgomezmiguez & mralarcon, Jan. 2020

  #Coefficients
  c_f00 = -2.2763;    c_f10 = 15.2827;    c_f20 = -197.789;
  c_f01 = -1.6850;    c_f11 = -9.2846;    c_f21 = 190.266;
  c_f02 = 0.76661;    c_f12 = 1.99381;    c_f22 = -67.9775;
  c_f03 = -0.053346;  c_f13 = -0.142631;  c_f23 = 10.6913;
  c_f04 = 0;          c_f14 = 0;          c_f24 = -0.625151;

  c_f = np.array([[c_f00, c_f01, c_f02, c_f03, c_f04],
                  [c_f10, c_f11, c_f12, c_f13, c_f14],
                  [c_f20, c_f21, c_f22, c_f23, c_f24]])

  b = np.arange(5)

  # Calculate lgwl matrix
  lgwl = np.log10(wl)**b.reshape(len(b),1)

  # Calculate f functions
  f_lgwl = np.einsum('ij,jk', c_f, lgwl)

  # Calculate sigma_ff_H_
  theta = 5040/T
  lgtheta = np.log10(theta)**np.arange(3)
  opacity_ff_H_ = 1e-26*n*Pe*10**(np.einsum('i,ij', lgtheta, f_lgwl))

  return opacity_ff_H_

def Opacity_bf(wl, n, sigma_bf, T):
# Function to measure bound-free opacity coefficient for hydrogen type atoms
# Input:
#       wl: array of wavelength in A
#        n: population of certain level
# sigma_bf: bound-free effective section
#        T: Temperature in K
# Output:
#  opacity_bf: bound-free opacity coefficient
# mmgomezmiguez & mralarcon, Jan. 2020
  opacity_bf = sigma_bf*n*(1 - np.e**(-h*c*1e8/(wl*K*T)))
  return opacity_bf

def Opacity_ff(wl, n, Ne, sigma_ff, T):
#function to measure free-free opacity coefficient for hydrogen type atoms
# Input:
#    wl: array of wavelength in A
#     n: population of certain level
#    Ne: density of free electrons
#  sigma_ff: free-free effective section
#  T: Temperature in K
# Output:
#  opacity_ff: free-free opacity coefficient
# mmgomezmiguez & mralarcon, Jan. 2020
  opacity_ff = sigma_ff*n*Ne*(1 - np.e**(-h*c*1e8/(wl*K*T)))
  return opacity_ff

def Opacity_Thom(Ne):
# Function to measure Thomson opacity coefficient
# Input:
#    Ne: density of free electrons
# Output:
#  k_Th: Thomson opacity coefficient
# mmgomezmiguez & mralarcon, Jan. 2020
  opacity_Th = 6.65e-25*Ne
  return opacity_Th


## --- OPACITY FUNCTION --- ##

def opacity_func(wl, atm, atom, filelist):
  # Number of depth points
  n_it = len(atm[filelist[0]]['T'])

  #Define opacity dictionary
  opacity_dict = {}
  for model in filelist:
    opacity_list = [[], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], []]
    for i in range(n_it):
      # H Gaunt factors
      g_bf_n1 = G_bf(wl, 1.0)
      g_bf_n2 = G_bf(wl, 2.0)
      g_bf_n3 = G_bf(wl, 3.0)
      g_bf_n4 = G_bf(wl, 4.0)

      g_ff = G_ff(wl, atm[model]['T'][i])

      g_bf_HeI = 1 # according to exercice notes

      # H- contribution
      sigma_bf_H_ = Sigma_bf_H_(wl, atom['H-'][1]['wlmaxA'])
      opacity_bf_H_ = Opacity_bf(wl, atm[model]['H-'][1.0]['N'][i], sigma_bf_H_
                                  , atm[model]['T'][i])
      opacity_list[0].append(opacity_bf_H_)

      opacity_ff_H_ = Opacity_ff_H_(wl, atm[model]['Pe'][i],
                                    atm[model]['HI']['N'][i],
                                    atm[model]['T'][i])
      opacity_list[1].append(opacity_ff_H_)

      #HI contribution
      sigma_bf_HI1 = Sigma_bf(wl, 1.0**2/R, 1,
                              1.0, g_bf_n1)
      sigma_bf_HI2 = Sigma_bf(wl, 2.0**2/R, 1,
                              2.0, g_bf_n2)
      sigma_bf_HI3 = Sigma_bf(wl, 3.0**2/R, 1,
                              3.0, g_bf_n3)
      sigma_bf_HI4 = Sigma_bf(wl, 4.0**2/R, 1,
                              4.0, g_bf_n4)

      opacity_bf_HI1 = Opacity_bf(wl, atm[model]['HI'][1.0]['N'][i],
                                  sigma_bf_HI1, atm[model]['T'][i])
      opacity_list[2].append(opacity_bf_HI1)

      opacity_bf_HI2 = Opacity_bf(wl, atm[model]['HI'][2.0]['N'][i],
                                  sigma_bf_HI2, atm[model]['T'][i])
      opacity_list[3].append(opacity_bf_HI2)

      opacity_bf_HI3 = Opacity_bf(wl, atm[model]['HI'][3.0]['N'][i],
                                  sigma_bf_HI3, atm[model]['T'][i])
      opacity_list[4].append(opacity_bf_HI3)

      opacity_bf_HI4 = Opacity_bf(wl, atm[model]['HI'][4.0]['N'][i],
                                  sigma_bf_HI4, atm[model]['T'][i])
      opacity_list[5].append(opacity_bf_HI4)

      # HII contribution
      sigma_ff_HII = Sigma_ff(wl, atm[model]['T'][i], 1, g_ff)

      opacity_ff_HII = Opacity_ff(wl, atm[model]['HII'][1.0]['N'][i],
                                atm[model]['Ne'][i],
                                sigma_ff_HII, atm[model]['T'][i])
      opacity_list[6].append(opacity_ff_HII)

      # HeI contribution
      sigma_bf_HeI1 = Sigma_HeI_n1(wl, atom['HeI'][1]['wlmaxA'])

      sigma_bf_HeI2 = Sigma_bf(wl, atom['HeI'][2]['wlmaxA'], 1,
                              2.0, g_bf_n2)
      sigma_bf_HeI2 *= 4*np.e**(-10.92*eV_to_erg/(K*atm[model]['T'][i]))  

      sigma_bf_HeI3 = Sigma_bf(wl, atom['HeI'][3]['wlmaxA'], 1,
                              2.0, g_bf_n2)
      sigma_bf_HeI3 *= 4*np.e**(-10.92*eV_to_erg/(K*atm[model]['T'][i]))  

      sigma_bf_HeI4 = Sigma_bf(wl, atom['HeI'][4]['wlmaxA'], 1,
                              2.0, g_bf_n2)
      sigma_bf_HeI4 *= 4*np.e**(-10.92*eV_to_erg/(K*atm[model]['T'][i])) 

      sigma_bf_HeI5 = Sigma_bf(wl, atom['HeI'][5]['wlmaxA'], 1,
                              2.0, g_bf_n2)
      sigma_bf_HeI5 *= 4*np.e**(-10.92*eV_to_erg/(K*atm[model]['T'][i])) 

      opacity_bf_HeI1 = Opacity_bf(wl, atm[model]['HeI'][1.0]['N'][i],
                                  sigma_bf_HeI1, atm[model]['T'][i])
      opacity_list[7].append(opacity_bf_HeI1)

      opacity_bf_HeI2 = Opacity_bf(wl, atm[model]['HeI'][2.0]['N'][i],
                                  sigma_bf_HeI2, atm[model]['T'][i])
      
      opacity_list[8].append(opacity_bf_HeI2)

      opacity_bf_HeI3 = Opacity_bf(wl, atm[model]['HeI'][3.0]['N'][i],
                                  sigma_bf_HeI3, atm[model]['T'][i])
      opacity_list[9].append(opacity_bf_HeI3)

      opacity_bf_HeI4 = Opacity_bf(wl, atm[model]['HeI'][4.0]['N'][i],
                                  sigma_bf_HeI4, atm[model]['T'][i])
      opacity_list[10].append(opacity_bf_HeI4)

      opacity_bf_HeI5 = Opacity_bf(wl, atm[model]['HeI'][5.0]['N'][i],
                                  sigma_bf_HeI5, atm[model]['T'][i])
      opacity_list[11].append(opacity_bf_HeI5)

      # HeII contribution
      sigma_bf_HeII1 = Sigma_bf(wl, atom['HeII'][1]['wlmaxA'], 2,
                                1.0, g_bf_HeI)
      # low value appears in n, so Python notices that we are dividing by zero.
      sigma_bf_HeII2 = Sigma_bf(wl, atom['HeII'][2]['wlmaxA'], 2,
                                2.0, g_bf_HeI)
      opacity_bf_HeII1 = Opacity_bf(wl, atm[model]['HeII'][1.0]['N'][i],
                                  sigma_bf_HeII1, atm[model]['T'][i])
      opacity_list[12].append(opacity_bf_HeII1)

      opacity_bf_HeII2 = Opacity_bf(wl, atm[model]['HeII'][2.0]['N'][i],
                                  sigma_bf_HeII2, atm[model]['T'][i])
      opacity_list[13].append(opacity_bf_HeII2)

      sigma_ff_HeII = sigma_ff_HII*4*np.e**(-10.92*eV_to_erg/(K*atm[model]['T'][i]))

      opacity_ff_HeII = Opacity_ff(wl, atm[model]['HeII'][1.0]['N'][i] + 
                                  atm[model]['HeII'][2.0]['N'][i],
                                  atm[model]['Ne'][i],
                                  sigma_ff_HeII, atm[model]['T'][i])
      opacity_list[14].append(opacity_ff_HeII)

      # HeIII contribution
      sigma_ff_HeIII = Sigma_ff(wl, atm[model]['T'][i], 2, g_bf_HeI)
      opacity_ff_HeIII = Opacity_ff(wl, atm[model]['HeIII'][1.0]['N'][i],
                                    atm[model]['Ne'][i],
                                    sigma_ff_HII, atm[model]['T'][i])
      opacity_list[15].append(opacity_ff_HeIII)

      # Free electron scattering
      opacity_Thom = Opacity_Thom(atm[model]['Ne'][i])
      opacity_list[16].append(opacity_Thom)

      # Total opacity
      opacity_tot = (opacity_bf_H_ + opacity_ff_H_ + opacity_bf_HI1 + 
                    opacity_bf_HI2 + opacity_bf_HI3 + opacity_bf_HI4 +
                    opacity_ff_HII + opacity_bf_HeI1 + opacity_bf_HeI2 +
                    opacity_bf_HeI3 + opacity_bf_HeI4 + opacity_bf_HeI5 +
                    opacity_bf_HeII1 + opacity_bf_HeII2 + opacity_ff_HeII +
                    opacity_ff_HeIII + opacity_Thom)
                  
      opacity_list[17].append(opacity_tot)
      
    # Save opacities in dictionary
    opacity_dict[model] = {'T'     : atm[model]['T'],
                           'lgtaur': atm[model]['lgtaur'],
                           'H-'    : {1.0 : {'bf' : opacity_list[0]},
                                      'ff' : opacity_list[1]},
                           'HI'    : {1.0 : {'bf' : opacity_list[2]},
                                      2.0 : {'bf' : opacity_list[3]},
                                      3.0 : {'bf' : opacity_list[4]},
                                      4.0 : {'bf' : opacity_list[5]}},
                           'HII'   : {'ff' : opacity_list[6]},
                           'HeI'   : {1.0 : {'bf' : opacity_list[7]},
                                      2.0 : {'bf' : opacity_list[8]},
                                      3.0 : {'bf' : opacity_list[9]},
                                      4.0 : {'bf' : opacity_list[10]},
                                      5.0 : {'bf' : opacity_list[11]}},
                           'HeII'  : {1.0 : {'bf' : opacity_list[12]},
                                      2.0 : {'bf' : opacity_list[13]},
                                      'ff' : opacity_list[14]},
                           'HeIII' : {'ff' : opacity_list[15]},
                           'Thom'  : opacity_list[16],
                           'tot'   : opacity_list[17]}
  return opacity_dict     