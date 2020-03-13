import supereeg as se
import numpy as np
import glob, os

# files = glob.glob('/dartfs/rc/lab/D/DBIC/CDL/f002s72/RAM_analysis/bos*.bo')
files = glob.glob('/dartfs/rc/lab/D/DBIC/CDL/f003f64/freqs/*.bo')

naned = []

for f in files:
    bo = se.load(f)
    numnans = np.isnan(bo.data.values).sum()
    if numnans > 0:
        print(f + ' has ' + str(numnans) + ' nans out of ' + str(bo.data.shape[0] * bo.data.shape[1]))
        naned.append('_'.join(os.path.basename(os.path.splitext(f)[0]).split('_')[:-1]))

print('final set')
print(set(naned))