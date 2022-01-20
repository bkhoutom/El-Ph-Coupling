'''
Bokang Hou
Jan 2022
Reorganization Energy for QD dimer
'''
# =============================================================================
# # IMPORT MODULES
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import argparse

# =============================================================================
# # CONSTANTS
# =============================================================================

pstoau = 41341.373335182114 # 1ps in atomic time units

autoev = 27.211324570273 # 1 atomic unit (Hartree) in eV
autoj = 4.3597447222071e-18 # 1 Hartree in J
evtoj = 1.602176634e-19 # 1 eV in J

autom = 5.291772083e-11 # 1 atomic distance unit in meters

kb_ev = 8.617333262145e-5 # Boltzmann constant (eV/K)
kb_au = kb_ev/autoev # Boltzmann constant (Hartree/K)
kb_j = kb_ev*evtoj # Boltzmann constant (J/K)


hbar = 1.054571817e-34 # J*s 


# =============================================================================
# # COMMAND LINE ARGUMENTS; Reading from input.par
# =============================================================================
def buildParser():

	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--nAtom', type=int, default='0',
			help='number of semiconductor atoms in quantum dot structure')
	parser.add_argument('--nElec', type=int, default='0',
			help='number of electrons used in BSE')
	parser.add_argument('--nHole', type=int, default='0',
			help='number of holes used in BSE')
	parser.add_argument('--temp', type=float, default='300',
			help='temperature (K)')
	return parser


# =============================================================================
# # IMPORT FILES
# =============================================================================
file1 = "ener_loc.dat"
file2 = "states_index.dat"
file3 = np.loadtxt("w.dat",delimiter=" ") # nu in Hz
file4 = np.loadtxt("Vkkq-diabatic-diag.dat",delimiter=" ")
lines = np.array([line.strip().split() for line in open(file1, 'r')])
index = np.array(lines)[:,0].astype(int) # index of the states based on increase of ener
ener = np.array(lines)[:,1].astype(float) # local states energies
LR = np.array([line.strip().split()[0] for line in open(file2, 'r')])[1:]
w = 2*np.pi*file3[:,1]*np.sqrt(1/evtoj)*10**(12) # w in the unit of sqrt(ev)/meter

# initialize index and energy(in eV) for n \in L and m \in R:
L_index = index[LR=="L"]
R_index = index[LR=="R"]
L_ener = ener[LR=="L"] * autoev
R_ener = ener[LR=="R"] * autoev

# =============================================================================
# # CALCULATION
# =============================================================================
# parameters
parser = buildParser()
params = parser.parse_args()

# initialize physical parameters:
T = params.temp # Temperature, K
beta = 1/(kb_ev*T)

# Partition functions for Left and Right states
Z_L = np.sum(np.exp(-beta*L_ener)) 
Z_R = np.sum(np.exp(-beta*R_ener)) 

# calculate \lambda_\alpha
lambda_alpha_arr = np.zeros(3*params.nAtom) # find how many alphas
V_kk_alpha = file4[:,-1]/evtoj # (in eV)

for alpha in range(3*params.nAtom):
    lambda_alpha = 0
    for n, nindex in enumerate(L_index):
        for m, mindex in enumerate(R_index):
            lambda_alpha += np.exp(-beta*(L_ener[n]+R_ener[m])) * 0.5 * (V_kk_alpha[alpha+nindex]-V_kk_alpha[alpha+mindex])**2/w[alpha]**2
            
    lambda_alpha_arr[alpha] = 1/(Z_L * Z_R) * lambda_alpha
    
            

np.savetxt("lambda.dat",lambda_alpha_arr)









    
