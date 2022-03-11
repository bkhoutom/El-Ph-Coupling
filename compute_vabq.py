'''
Bokang Hou modified from
Dipti Jasrasaria
Jan 2022

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORT MODULES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''

import os
import numpy as np
import scipy.linalg as la
import argparse
'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONSTANTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
pstoau = 41341.373335182114 # 1ps in atomic time units

autoev = 27.211324570273 # 1 atomic unit (Hartree) in eV
autoj = 4.3597447222071e-18 # 1 Hartree in J
evtoj = 1.602176634e-19 # 1 eV in J

autom = 5.291772083e-11 # 1 atomic distance unit in meters

kb_ev = 8.617333262145e-5 # Boltzmann constant (eV*K)
kb_au = kb_ev/autoev # Boltzmann constant (Hartree*K)
kb_j = kb_ev*evtoj # Boltzmann constant (J*K)

hbar = 1.054571817e-34 # J*s 

'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# COMMAND LINE ARGUMENTS; Reading from input.par
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
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

'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# READ INPUT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Construct Hamiltonian Hsys_tot and Hsys_tot_ucp

def construct_ham_n_U(nElec,nHole):
    filenames = ['umat.par', 'hmat.par','states_index.par']
    for filename in filenames:
        if not os.path.exists(filename):
            print('Cannot find ' + filename + '! Exiting...\n')
            exit()
    if nElec * nHole != 0:
        print("Must input either electrons or holes, can't do mixed...")
        exit()
    if nElec != 0:
        nQPs = nElec
    elif nHole != 0:
        nQPs = nHole
    else:
        print("Error in the number of nElec or nHole")
        exit()
    
    LR = np.array([line.strip().split()[0] for line in open("states_index.par", 'r')])[1:]
    index = np.array([line.strip().split()[1] for line in open("states_index.par", 'r')])[1:].astype(int)
    L_index = index[LR=="L"]
    R_index = index[LR=="R"]
    umat = np.loadtxt("umat.par",delimiter=' ')
    U = umat[:,-1].reshape((nQPs,nQPs)) # localization transform U
    hmat = np.loadtxt("hmat.par",delimiter=' ')
    Hsys_tot = hmat[:,-1].reshape((nQPs,nQPs)) # H_DA Hamiltonian from local
    np.savetxt("H_sys_tot.csv",Hsys_tot*autoev,delimiter=",")
    
    # construct DA uncoupled Hamil.
    H0 = Hsys_tot.copy()
    H0[len(L_index):,0:len(L_index)] = 0
    H0[0:len(L_index),len(L_index):] = 0
    E0, U0 = la.eig(H0)
    Hsys_tot_ucp = U0.T.conjugate() @ Hsys_tot @ U0
    np.savetxt("H_sys_tot_ucp.csv",Hsys_tot_ucp*autoev,delimiter=",")
    
    # construct total unitary tranform Utlt
    Utlt = U @ U0
    np.savetxt("umat_tlt.dat",Utlt)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# read in  matrix element U_na from umat.dat from localization

def read_umat(nElec, nHole):
    filename = 'umat_tlt.dat' ##############to umat_tilt.par
    if not os.path.exists(filename):
        print('Cannot find ' + filename + '! Exiting...\n')
        exit()
    if nElec * nHole != 0:
        print("Must input either electrons or holes, can't do mixed...")
        exit()
    umat = np.loadtxt("umat.par",delimiter=' ')
    if nElec != 0:
        nQPs = nElec
    elif nHole != 0:
        nQPs = nHole
    else:
        print("Error in the number of nElec or nHole")
        exit()
    Utlt = umat[:,-1].reshape((nQPs,nQPs))
    return Utlt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # read masses of semiconductor atoms from conf.par
def read_mass(natoms):
    filename = 'conf.par'
    if not os.path.exists(filename):
        print('Cannot find ' + filename + '! Exiting...\n')
        exit()

    lines = np.array([line.strip().split() for line in open(filename, 'r')])[1:]
    at_type = np.array([line[0] for line in lines])[:natoms]
    
    # make sure no passivation atoms
    if ('P1' == at_type).any() or ('P2' == at_type).any():
        print('Error reading ' + filename + '! Exiting...\n')
        exit()

    mass = np.empty(natoms, dtype=np.float)
    cd_idx = np.where(at_type == 'Cd')[0]
    se_idx = np.where(at_type == 'Se')[0]
    s_idx = np.where(at_type == 'S')[0]
    if (cd_idx.shape[0] + se_idx.shape[0] + s_idx.shape[0] != natoms):
        print('Error reading ' + filename + '! Exiting...\n')
        exit()
    mass[cd_idx] = 112.411
    mass[se_idx] = 78.960
    mass[s_idx] = 32.065
    mass *= (1e-3/6.022140857e23) # kg

    return mass

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # read mass-weighted atomic positions of semiconductor atoms from conf.par
def read_pos(natoms):
    filename = 'conf.par'
    if not os.path.exists(filename):
        print('Cannot find ' + filename + '! Exiting...\n')
        exit()

    lines = np.array([line.strip().split() for line in open(filename, 'r')])[1:]
    at_type = np.array([line[0] for line in lines])[:natoms]
    
    # make sure no passivation atoms
    if ('P1' == at_type).any() or ('P2' == at_type).any():
        print('Error reading ' + filename + '! Exiting...\n')
        exit()
        
    #print(lines)
    mass = np.empty(natoms, dtype=np.float)
    cd_idx = np.where(at_type == 'Cd')[0]
    se_idx = np.where(at_type == 'Se')[0]
    s_idx = np.where(at_type == 'S')[0]
    if (cd_idx.shape[0] + se_idx.shape[0] + s_idx.shape[0] != natoms):
        print('Error reading ' + filename + '! Exiting...\n')
        exit()
    mass[cd_idx] = 112.411
    mass[se_idx] = 78.960
    mass[s_idx] = 32.065
    mass *= (1e-3/6.022140857e23) # kg
    pos = np.zeros((natoms,3))
    for count, line in enumerate(lines[:natoms]):
        pos[count] = np.array(lines[count])[1:].astype(np.float) # dirty way of change datatype (unit:meter)
    pos_mass = pos * np.sqrt(mass[:, None]) * 10**(-10) # convert anstrogm to meter
    return pos_mass

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # read in phonon modes from eig.dat
# i.e. eigenvectors of dynamical matrix to transform to phonon mode coordinates
# q = U*r

def read_phononModes(natoms):
    filename = 'eig.dat'
    if not os.path.exists(filename):
        print('Cannot find ' + filename + '! Exiting...\n')
        exit()

    # pmodes = np.array([line.strip().split() for line in open(filename, 'r')]).astype(np.float).transpose()#########bug: memory error##########
    # print(pmodes)
    pmodes = np.fromfile('eig.dat',dtype=float,sep=" ").reshape((3*natoms,3*natoms)).T


    if (pmodes.shape[0] != 3*natoms) or (pmodes.shape[1] != 3*natoms):
        print ('Error reading ' + filename + '! Exiting...\n')
        exit()

    return pmodes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # read in nonadiabatic coupling matrix elements from Vab-diabatic.dat or Vij-diabatic.dat
def read_DiabCoupling(natoms, nQPs, Vrs_file):
    """
    natoms: int, number of semiconductor atoms
    nQPs: int, number of either electrons or holes
    Vrs_file: FILE, either vab_uk or vij_uk
    """
    if not os.path.exists(Vrs_file):
        print('Cannot find ' + Vrs_file + '! Exiting...\n')
        exit()

    lines = np.array([line.strip().split() for line in open(Vrs_file, 'r')])
    qp1Idx = np.array([line[6] for line in lines]).astype(np.int)
    qp2Idx = np.array([line[8] for line in lines]).astype(np.int)
    lines = lines[np.where((qp1Idx < nQPs) & (qp2Idx < nQPs))[0]][:nQPs*nQPs*natoms]

    if len(lines) != (natoms*nQPs*nQPs):
        print('Error reading in ' + Vrs_file + '! Exiting...Check the dimension of config file\n')
        exit()
    
    Vrsx = np.array([line[10] for line in lines]).astype(np.float).reshape((natoms, nQPs, nQPs))
    Vrsy = np.array([line[11] for line in lines]).astype(np.float).reshape((natoms, nQPs, nQPs))
    Vrsz = np.array([line[12] for line in lines]).astype(np.float).reshape((natoms, nQPs, nQPs))
    Vrs = np.zeros((3*natoms, nQPs, nQPs))
    Vrs[::3,:,:] = Vrsx
    Vrs[1::3,:,:] = Vrsy
    Vrs[2::3,:,:] = Vrsz

    return Vrs

'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# COMPUTE ELECTRON-PHONON COUPLING
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
def compute_eleCouplingAtomic(natoms, nElec, nHole):

    idx = 0
    nQPs = nElec + nHole
    # Read in quasiparticle diabatic couplings
    if nElec != 0:
        Vkl = np.zeros((3*natoms, nElec, nElec))
        Vab = read_DiabCoupling(natoms, nElec, 'Vab-diabatic.dat')
    elif nHole != 0:
        Vkl = np.zeros((3*natoms, nHole, nHole))
        Vab = read_DiabCoupling(natoms, nHole, 'Vij-diabatic.dat')

    # Unitary tranform between psi and phi
    Utlt = read_umat(nElec,nHole)
    
    # electron or hole contribution

    for n in range(nQPs):
        Vnn = np.zeros(np.shape(Vab[:,0,0]))
        for a in range(nQPs):
            for b in range(nQPs):
                 Vnn += Utlt[n,a]*Utlt[n,b]*Vab[:,a,b]
        Vkl[:,n,n] = Vnn

    Vkl *= autoj/autom # J / m
    with open('Vkl-diabatic.dat', 'w') as f:
        for n in range(Vkl.shape[0]):
            for i in range(Vkl.shape[1]):
                for j in range(Vkl.shape[2]):
                    f.write(str(n) + ' ' + str(i) + ' ' + str(j) + ' ' + str(Vkl[n,i,j]) + '\n')
    return Vkl



def compute_eleCouplingPhonon(natoms, nElec, nHole):

    # Vkl in atomic coordinates ( J / m)
    Vkl = compute_eleCouplingAtomic(natoms, nElec, nHole)
    nQPs = nElec + nHole
    # phonon modes
    pmodes = read_phononModes(natoms)
    pmodes_inv = pmodes.transpose()
    sr_mass3 = np.sqrt(np.repeat(read_mass(natoms), 3)) # sqrt(kg)

    # compute Vklq
    if nElec != 0:
        Vklq = np.empty((3*natoms, nElec, nElec))
    elif nHole != 0:
        Vklq = np.empty((3*natoms, nHole, nHole))

    for k in range(Vkl.shape[1]):
        for l in range(Vkl.shape[2]):
            Vklq[:,k,l] = np.matmul(pmodes_inv, Vkl[:,k,l]/sr_mass3) # J / sqrt(kg)*m

    with open('Vklq-diabatic.dat', 'w') as f1, open('Vkkq-diabatic-diag.dat', 'w') as f2:
        for n in range(Vklq.shape[0]):
            for i in range(Vklq.shape[1]):
                for j in range(Vklq.shape[2]):
                    f1.write(str(n) + ' ' + str(i) + ' ' + str(j) + ' ' + str(Vklq[n,i,j]) + '\n')
                    if i==j:
                        f2.write(str(n) + ' ' + str(i) + ' ' + str(j) + ' ' + str(Vklq[n,i,j]) + '\n')

    return Vklq





# # # # parameters
parser = buildParser()
params = parser.parse_args()
construct_ham_n_U(params.nElec,params.nHole)
# compute_eleCouplingAtomic(params.nAtom,params.nElec,params.nHole)
vklq = compute_eleCouplingPhonon(params.nAtom,params.nElec,params.nHole)
#print(vklq)
