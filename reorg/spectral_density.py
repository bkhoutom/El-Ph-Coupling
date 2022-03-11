import numpy as np
import matplotlib.pyplot as plt

# load data
f = np.array([line.strip().split()[0] for line in open('3.9nmCdSe_reorg0.dat', 'r')]).astype(np.float) # frequency (THz)
r = np.array([line.strip().split()[-1] for line in open('3.9nmCdSe_reorg0.dat', 'r')]).astype(np.float) # reorganization energy (meV)
# hr = np.array([line.strip().split()[-1] for line in open('hr-avg.dat', 'r')]).astype(np.float) # huang-rhys factor

# broadening
gauss_sigma = 0.05
gauss_prefactor = 1./(gauss_sigma*np.sqrt(2.*np.pi))

# binning
f_bins = np.arange(0., 7.8, 0.001)
df = f_bins[1]-f_bins[0]

spectral_density_r = np.zeros_like(f_bins)
# spectral_density_hr = np.zeros_like(f_bins)

# bin reorganization energies by frequency (i.e. spectral density)
for i in range(f.shape[0]):
    idx = np.argmin(np.abs(f[i]-f_bins))
    spectral_density_r += (r[i]*gauss_prefactor*np.exp(-0.5*((f_bins-f[i])/gauss_sigma)**2))
   # spectral_density_hr += (hr[i]*gauss_prefactor*np.exp(-0.5*((f_bins-f[i])/gauss_sigma)**2))

plt.clf()
plt.plot(f_bins, spectral_density_r)
plt.ylim([0., 30.])
plt.axvline(6.04497)
plt.title('3.9nmcdse')
plt.savefig('3.9nmcdse_jw.png')
exit()
plt.show()

plt.clf()
plt.plot(f_bins, spectral_density_r)
plt.yscale('log')
plt.show()

'''
plt.clf()
plt.plot(f_bins, spectral_density_hr)
plt.show()

plt.clf()
plt.plot(f_bins, spectral_density_hr)
plt.yscale('log')
plt.show()
'''
