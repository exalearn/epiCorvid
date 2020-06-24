import numpy as np
import h5py
import matplotlib.pyplot as plt


fname ='/global/cfs/cdirs/m3623/datasets/RLtest/data.h5'
with h5py.File(fname, 'r') as f:
    dat = f['symptomatic3D'][...]
    bio = f['bioPar']
    uni = f['uniPar']


hbins = np.arange(-0.5,134.5,1)
centers = np.arange(134)
print(hbins, centers)

hist, _ = np.histogram(dat, bins=hbins)

plt.figure()
plt.plot(centers, hist, 'o')
plt.yscale('log')
plt.xlabel('Num. new infected per age/day/tract')
plt.ylabel('Counts')
plt.title('symptomatic3D')
plt.show()



