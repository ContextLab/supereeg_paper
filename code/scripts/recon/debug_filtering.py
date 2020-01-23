import supereeg as se
import os, glob
from config import config
from bandbrain import BandBrain
from supereeg.load import load
import pandas as pd


files = glob.glob(os.path.join(config['og_bodir'], '*'))
fullmatdict = dict()
def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)

for f in files:
    num_elecs = electrode_search(f)
    fullmatdict[f] = num_elecs

def filter_subj(bo, measure='kurtosis', return_locs=False, threshold=10):
    locs = load(bo, field='locs')
    kurt_vals = load(bo, field='kurtosis')
    meta = load(bo, field='meta')
    thresh_bool = kurt_vals > threshold
    if sum(~thresh_bool) < 2:
        pass
    else:
        if return_locs:
            locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
            return meta, locs[~thresh_bool]
        else:
            return meta


def electrode_search2(fname):
    values = filter_subj(fname, return_locs=True)
    if values is None:
        return 0
    else:
        meta, locs = values
        return locs.shape[0]
    

recondict = dict()
for f in files:
    num_elecs = electrode_search2(f)
    recondict[f] = num_elecs

diffdict = dict()
for k in recondict.keys():
    diffdict[k] = - recondict[k] + fullmatdict[k]


fname = glob.glob(os.path.join(config['datadir'], '*'))[0]
freq = os.path.basename(fname).split('_')[-1].split('.bo')[0]
og_fname = os.path.join(config['og_bodir'], os.path.basename(fname).split('_'+freq)[0] + '.bo')
bo = se.load(fname)
og_bo = se.load(og_fname)
band = BandBrain(bo, og_bo)