#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esto código permite obtener las poblaciones de los niveles de excitación de
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


def partition(g_j, X_j, T):
# Returns the partition function U_j of the ionization state
# Input:
#   g_j: array of statistical weights of the levels in the ionization state j
#   X_j: array of excitation energies of the levels in the ionization state j (in eV)
#     T: temperature (in K)
# Output:
#   U_j: partition function of the ionization state j (in J)
# mralarcon & mmgomezmiguez, Dec. 2019

  # Convert excitation energy from eV to erg
  X_j = X_j * eV_to_erg

  # Sum over all the excitation levels in the ionization state
  U_j = np.sum(g_j * np.exp(-X_j/(K*T)))

  return U_j


def saha(Ne, N_j1, U_j, U_j1, I_j, T):
# Returns the relative populations of the element in different ionization
# mralarcon & mmgomezmiguez, Dec. 2019

  # Convert ionization energy from eV to erg
  I_j = I_j * eV_to_erg

  # Saha equation
  N_j = N_j1 * 2.07e-16 * Ne * U_j/U_j1 * T**(-3/2) * np.exp(I_j/(K*T))

  return N_j


def boltzmann(N_j, g_j, U_j, X_j, T):
# Returns the relative populations of the ionization state j
# mralarcon & mmgomezmiguez, Dec. 2019

  # Convert excitation energy from eV to erg
  X_j = X_j * eV_to_erg

  # Boltzmann equation
  N_ji = N_j * g_j/U_j * np.exp(-X_j/(K*T))

  return N_ji


def populations(atom, model, abunH, abunHe):

  # Number of free electrons
  lgtau, T, Pe = model[0], model[2], model[3]
  Ne = Pe / (K * T)

  # Partition fuctions
  UH_ = partition(atom['H-']['g'], atom['H-']['X'], T)
  UHI = partition(atom['HI']['g'], atom['HI']['X'], T)
  UHII = partition(atom['HII']['g'], atom['HII']['X'], T)
  UHeI = partition(atom['HeI']['g'], atom['HeI']['X'], T)
  UHeII = partition(atom['HeII']['g'], atom['HeII']['X'], T)
  UHeIII = partition(atom['HeIII']['g'], atom['HeIII']['X'], T)

  # We obtain the equations in matricial form: B=MxA,
  # with A = [NH-, NHI, NHII, NHeI, NHeII, NHeIII] and
  B = np.array([0, 0, 0, 0, Ne, 0])

  # 4 saha equations
  M1 = np.array([-1, saha(Ne, 1, UH_, UHI, atom['H-']['I'], T), 0, 0, 0, 0])
  M2 = np.array([0, -1, saha(Ne, 1, UHI, UHII, atom['HI']['I'], T), 0, 0, 0])
  M3 = np.array([0, 0, 0, -1, saha(Ne, 1, UHeI, UHeII, atom['HeI']['I'], T), 0])
  M4 = np.array([0, 0, 0, 0, -1, saha(Ne, 1, UHeII, UHeIII, atom['HeII']['I'], T)])

  # charge conservation
  M5 = np.array([-1, 0, 1, 0, 1, 2])

  # relative abundances
  abun_dif = 10**(abunH - abunHe)
  M6 = np.array([1, 1, 1, -abun_dif, -abun_dif, -abun_dif])

  # solve equation system
  M = np.vstack((M1, M2, M3, M4, M5, M6))
  N = np.linalg.solve(M, B)

  # get the populations by excitation levels
  n = []
  for i,ie in enumerate(atom):
    U = partition(atom[ie]['g'], atom[ie]['X'], T)
    nie = boltzmann(N[i], atom[ie]['g'], U, atom[ie]['X'], T)
    n.append(nie)
    # check
    if np.sum(nie) > N[i]: 'Check your populations!'
  n = np.array(n)

  return N, Ne, n