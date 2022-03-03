# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:40:24 2021

@author: HBK
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# =============================================================================
# Phonon DOS
# =============================================================================
# Creating dataset
n_bins = 150
 
# Creating distribution
data_omega = np.loadtxt("w.dat",delimiter=' ')
w = data_omega[:,1]
 
# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
 
dos, freq, patches = axs.hist(w, bins = n_bins)

# Show plot
plt.xlabel("Frequency (THz)")
plt.ylabel("Phonon DOS")
#plt.savefig("dimer_ph_dos.pdf")
plt.show()

# =============================================================================
# Electron-phonon coupling Matrix V_nn^alpha
# =============================================================================

# plt.rcParams.update({'font.size': 10})
# data_vklq = np.loadtxt("Vklq-diabatic.dat",delimiter=' ')
# vklq_eV = data_vklq[:,-1] * 6.241509074460763e+18
# plt.plot(vklq_eV,'o',markersize=3,color='C1')
# plt.xlabel(r"(n,$\alpha$)")
# plt.ylabel(r"$V_{nm}^\alpha$ (eV/m)")
# plt.savefig('elphCP_Vnm^alpha.pdf')
# plt.show()


# =============================================================================
# Plot the spectral density J(w)
# =============================================================================

lambda_alpha = np.loadtxt("lambda.dat",delimiter=" ")
lambda_tot = np.sum(lambda_alpha[5:])
plt.plot(w[0:],lambda_alpha[0:],'o')
plt.yscale("log")

lam_ave = []
for i in range(n_bins):
    ave = np.average(lambda_alpha[i*len(w)//n_bins:(i+1)*len(w)//n_bins])
    lam_ave.append(ave)

plt.plot(freq[1:],lam_ave)
plt.show()


J_w = dos*np.array(lam_ave)
I1 = integrate.simps(J_w[1:], freq[2:])
plt.plot(freq[1:],J_w/I1*lambda_tot,'-')
plt.xlabel("Frequency (THz)")
plt.ylabel(r"$J(\omega)$")
plt.savefig("J_omega.pdf")



