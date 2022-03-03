# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:50:13 2021

@author: HBK
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 25})

###############################Plot eigen and local states energies###########
##############################################################################
autoev = 27.211324570273 # 1 atomic unit (Hartree) in eV
#eval_e = np.loadtxt('eval.dat')
ener_l = np.loadtxt('ener_loc.dat')[:,1]
#eig_e = np.sort(eval_e[360:370,1]) * autoev
#loc_e = np.sort(ener_l[:,1]) * autoev


def diagram(energy,color='k',style=None,alpha=1,label=None):
    
    plt.plot([-0.3,+0.3],[energy[0],energy[0]],color=color,linewidth=5,linestyle=style,alpha=alpha,label=label)
    for index, e in enumerate(energy[1:]):
        plt.plot([index+1-0.3,index+1+0.3],[e,e],color=color,linewidth=5,linestyle=style,alpha=alpha)
        
    
    plt.xticks(np.arange(len(energy)))
    return 0

#plt.figure(figsize=(60,13))
#plt.figure(figsize=(30,13))
#diagram(loc_e,label='localized states') #Using 'last'  or 'l' it will be together with the previous level
#diagram(eig_e,color='r',style='--',alpha=0.3,label='eigenstates')
#plt.ylim([-0.26,-0.14])
# plt.ylim([-4.6,-4.0])
# plt.xlabel("Orbital Label")
# plt.ylabel("Energy (eV)")
# plt.legend(loc='upper left')
#plt.savefig('eigen_local_10e.pdf')
#plt.savefig('eigen_10e.pdf')
# plt.savefig('local_10e.pdf')
# plt.show()



########################Plot Jnm couplings#####################################
###############################################################################
plt.rcParams.update({'font.size': 10})
data_J = np.loadtxt("hmat.dat",delimiter=' ')

LR = np.array([line.strip().split()[0] for line in open("states_index.dat", 'r')])[1:]
index = np.array([line.strip().split()[1] for line in open("states_index.dat", 'r')])[1:].astype(int)
L_index = index[LR=="L"]
R_index = index[LR=="R"]
L_ener = ener_l[LR=="L"] * autoev
R_ener = ener_l[LR=="R"] * autoev



Jnm = data_J[:,-1]* autoev
plt.plot(Jnm,'o',markersize=3)
plt.ylim([-0.08,0.08])
plt.xlabel("(n,m)")
plt.ylabel(r"$J_{nm}$ (eV)")
plt.savefig('electronicCP_Jnm.pdf')
plt.show()
nm = 0
states_num=8
for n in L_index:
    for m in R_index:
        plt.plot(nm,Jnm[states_num*n+m],"o",color="C1",markersize=3)
        nm +=1
plt.ylim([-0.03,0.03])
plt.xlabel(r"($n\in D$, $m\in A$)")
plt.ylabel(r"$J_{nm}$ (eV)")
plt.savefig('electronicCP_Jnm_DA.pdf')
plt.show()

# =============================================================================
# Plot Delta_E for n\in D and m\in A
# =============================================================================
delE = np.zeros(len(L_index)*len(R_index))
i=0
for n, nindex in enumerate(L_index):
    for m, mindex in enumerate(R_index):
        delE[i] = np.abs(L_ener[n]-R_ener[m])
        i+=1
plt.plot(delE,"o",color="C1")
plt.xlabel(r"($n\in D$, $m\in A$)")
plt.ylabel(r"$\Delta E_{nm}$ (eV)")
plt.ylim([0,0.014])
plt.savefig("delta_E.pdf")
