
import supereeg as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import pickle
from config import config
from stats import time_by_file_index_chunked_local, corrmat, z2r, r2z
from bookkeeping import remove_electrode, known_unknown, alter_avemat



## load brain object
bo_fname = sys.argv[1]
bo = se.load(bo_fname)

file_name = os.path.basename(os.path.splitext(bo_fname)[0])

npz_infile = sys.argv[1][:-2] + 'npz'
sub_data = np.load(npz_infile)

elec_ind = int(sys.argv[2])

model_template = sys.argv[3]

mo_fname = os.path.join(config['modeldir'], model_template, file_name + '.npz')
mo = np.load(mo_fname, mmap_mode='r')


ave_dir = os.path.join(config['avedir'], model_template)

results_dir = os.path.join(config['resultsdir'], model_template)


try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')

bo.get_filtered_bo()

R = np.load(locs_file)['locs']

Ave_data = np.load(os.path.join(ave_dir, 'locs.npz'), mmap_mode='r')

Model, count = alter_avemat(Ave_data, mo)


Model[np.where(np.isnan(Model))] = 0
Model = Model + np.eye(np.shape(Model)[0])

## subject's locations
R_K_subj = bo.get_locs().as_matrix()

## get info for held out electrode
electrode = bo.locs.iloc[elec_ind]

## remove electrode and get indices
R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)

## inds after kurtosis threshold: known_inds = known electrodes; unknown_inds = all the rest; rm_unknown_ind = where the removed electrode is located in unknown subset
known_inds, unknown_inds, electrode_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)

## get correlation (if timeseries=True, returns the timeseries instead)
corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                            elec_ind, time_series=False)

recon_outfile = os.path.join(results_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
## save each file
np.savez(recon_outfile, coord=electrode, corrs=corrs)

