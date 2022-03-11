import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('w_file', type=str)
parser.add_argument('eig_file', type=str)
parser.add_argument('nzero', type=int) # number of zero modes
parser.add_argument('Vaa_file', type=str) # electron states
parser.add_argument('Vii_file', type=str) # hole states
parser.add_argument('Vkke_file', type=str) # electron contribution to excitonic states
parser.add_argument('Vkkh_file', type=str) # hole contribution to excitonic states
parser.add_argument('nexc', type=int) # number of corresponding excitonic state
args = parser.parse_args()

# # # # constants # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
autoj = 4.3597447222071e-18 # 1 Hartree in J
autom = 5.291772083e-11 # 1 atomic distance unit in meters
hbar = 1.054571817e-34 # J*s 
evtoj = 1.602176634e-19 # 1 eV in J

# # # # phonons # # # # # # # # # # # # # # # # # # # # # # # # # # #
# note: this is omega (i.e. angular frequency)
freq = np.array([line.strip().split()[1] for line in open(args.w_file, 'r')]).astype(np.float)*1e12*2.*np.pi # 1 / s
# matrix to transform to phonon mode coordinates (i.e. q = U*r)
eig = np.array([line.strip().split() for line in open(args.eig_file, 'r')]).astype(np.float).transpose()
# matrix to transform to atomic coordinates (i.e. r = Ut*q)
eiginv = eig.transpose()

#print('Finished reading phonon modes...\n')

# # # # parameters # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# number of dimensions
ndim = freq.shape[0]
# number of atoms
natoms = ndim/3

# # # # el-ph couplings # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# units are au / au (energy / distance)
# electrons
lines = np.array([line.strip().split() for line in open(args.Vaa_file, 'r')])
if lines.shape[0] != natoms:
    print('Error reading ' + args.Vaa_file + '! Exiting...\n')
    exit()
at_idx = np.array([line[0] for line in lines]).astype(np.int)
nax = np.array([line[10] for line in lines]).astype(np.float)
nay = np.array([line[11] for line in lines]).astype(np.float)
naz = np.array([line[12] for line in lines]).astype(np.float)

na_nie = np.dstack((nax, nay, naz)).flatten()
na_nie *= (autoj/autom) # J / m

# holes
lines = np.array([line.strip().split() for line in open(args.Vii_file, 'r')])
if lines.shape[0] != natoms:
    print('Error reading ' + args.Vii_file + '! Exiting...\n')
    exit()
at_idx = np.array([line[0] for line in lines]).astype(np.int)
nax = np.array([line[10] for line in lines]).astype(np.float)
nay = np.array([line[11] for line in lines]).astype(np.float)
naz = np.array([line[12] for line in lines]).astype(np.float)

na_nih = np.dstack((nax, nay, naz)).flatten()
na_nih *= (autoj/autom) # J / m

na_ni = na_nie - na_nih

# excitons
liness = np.array([line.strip().split() for line in open(args.Vkke_file, 'r')])
if liness.shape[0] != ndim:
    print('Error reading ' + args.Vkke_file + '! Exiting...\n')
    exit()
at_idx = np.array([line[0] for line in liness]).astype(np.int)
na_bse = np.array([line[-1] for line in liness]).astype(np.float)
na_bse *= (autoj/autom) # J / m

liness = np.array([line.strip().split() for line in open(args.Vkkh_file, 'r')])
if liness.shape[0] != ndim:
    print('Error reading ' + args.Vkkh_file + '! Exiting...\n')
    exit()
at_idx = np.array([line[0] for line in liness]).astype(np.int)
na_bsh = np.array([line[-1] for line in liness]).astype(np.float)
na_bsh *= (autoj/autom) # J / m

na_bs = na_bse - na_bsh

#print('Finished reading electron-phonon coupling...\n')

# # # # mass # # # # # # # # # # # # # # # # # # # # # # # # # # #
# mass (kg)
at_type = np.array([line[1] for line in lines])
mass = np.empty(natoms, dtype=np.float)
mass[np.where(at_type == 'Cd')[0]] = 112.411
mass[np.where(at_type == 'Se')[0]] = 78.960
mass[np.where(at_type == 'S')[0]] = 32.065
mass *= (1e-3/6.022140857e23)
mass3 = np.repeat(mass, 3)

# el-ph coupling in normal coordinates (i.e. dH/dq) with units J / m*sqrt(kg)
naqnie = np.matmul(eiginv, na_nie/np.sqrt(mass3))
naqnih = np.matmul(eiginv, na_nih/np.sqrt(mass3))
naqni = np.matmul(eiginv, na_ni/np.sqrt(mass3))
naqbse = np.matmul(eiginv, na_bse/np.sqrt(mass3))
naqbsh = np.matmul(eiginv, na_bsh/np.sqrt(mass3))
naqbs = np.matmul(eiginv, na_bs/np.sqrt(mass3))

# reorganization energy in J
n = args.nzero
reorg_nie = 0.5*((naqnie[n:])/freq[n:])**2 
reorg_nih = 0.5*((naqnih[n:])/freq[n:])**2 
reorg_ni = 0.5*((naqni[n:])/freq[n:])**2
reorg_bse = 0.5*((naqbse[n:])/freq[n:])**2 
reorg_bsh = 0.5*((naqbsh[n:])/freq[n:])**2 
reorg_bs = 0.5*((naqbs[n:])/freq[n:])**2 

# Huang Rhys parammeter (dimensionless)
hr_nie = reorg_nie/(hbar*freq[n:])
hr_nih = reorg_nih/(hbar*freq[n:])
hr_ni = reorg_ni/(hbar*freq[n:])
hr_bse = reorg_bse/(hbar*freq[n:])
hr_bsh = reorg_bsh/(hbar*freq[n:])
hr_bs = reorg_bs/(hbar*freq[n:])

#print('ni-cbm reorg_e (eV): ', np.sum(reorg_nie)/evtoj)
print 'bs-cbm reorg_e (eV): ', np.sum(reorg_bse)/evtoj

#print('ni-vbm reorg_e (eV): ', np.sum(reorg_nih)/evtoj)
print 'bs-vbm reorg_e (eV): ', np.sum(reorg_bsh)/evtoj

#print('ni-exc reorg_e (eV): ', np.sum(reorg_ni)/evtoj)
print 'bs-exc reorg_e (eV): ', np.sum(reorg_bs)/evtoj

freq *= (1e-12/(2.*np.pi)) # convert back to THz
reorg_nie *= (1000./evtoj) # convert to meV
reorg_nih *= (1000./evtoj) # convert to meV
reorg_ni *= (1000./evtoj) # convert to meV
reorg_bse *= (1000./evtoj) # convert to meV
reorg_bsh *= (1000./evtoj) # convert to meV
reorg_bs *= (1000./evtoj) # convert to meV

with open('reorg-' + str(args.nexc) + '.dat', 'w') as f:
    for i in range(freq.shape[0]-n):
        f.write(str(freq[i+n]) + ' ')
        f.write(str(reorg_nie[i]) + ' ')
        f.write(str(reorg_nih[i]) + ' ')
        f.write(str(reorg_ni[i]) + ' ')
        f.write(str(reorg_bse[i]) + ' ')
        f.write(str(reorg_bsh[i]) + ' ')
        f.write(str(reorg_bs[i]) + '\n')

with open('hr-' + str(args.nexc) + '.dat', 'w') as f:
    for i in range(freq.shape[0]-n):
        f.write(str(freq[i+n]) + ' ')
        f.write(str(hr_nie[i]) + ' ')
        f.write(str(hr_nih[i]) + ' ')
        f.write(str(hr_ni[i]) + ' ')
        f.write(str(hr_bse[i]) + ' ')
        f.write(str(hr_bsh[i]) + ' ')
        f.write(str(hr_bs[i]) + '\n')
