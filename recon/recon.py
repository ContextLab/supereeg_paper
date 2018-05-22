
import supereeg as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import pickle
from config import config
from supereeg.helpers import filter_subj



## load brain object
bo_fname = sys.argv[1]
bo = se.load(bo_fname)

file_name = os.path.basename(os.path.splitext(bo_fname)[0])

if bo_fname.split('.')[-1]=='bo':
    bo = se.load(bo_fname)
    npz_infile = sys.argv[1][:-2] + 'npz'
    sub_data = np.load(npz_infile)
    mo = os.path.join(config['modeldir'], file_name + '.npz')

elec_ind = int(sys.argv[2])

model_template = sys.argv[3]

ave_dir = os.path.join(config['avedir'], model_template)

results_dir = os.path.join(config['resultsdir'], model_template)

locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')

bo.get_filtered_bo()

R = np.load(locs_file)['locs']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

Ave_data = np.load(os.path.join(ave_dir, 'ave_model_full_model.npz'), mmap_mode='r')



def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)


