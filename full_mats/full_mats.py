
import supereeg as se
from supereeg.model import _bo2model
import numpy as np
import glob
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

radius = sys.argv[3]

results_dir = os.path.join(config['resultsdir'], model_template +"_"+ radius)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)

locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')

R = np.load(locs_file)['locs']

elec_count = electrode_search(sys.argv[1])

if elec_count > 1:

    fname =os.path.basename(os.path.splitext(fname)[0])

    print('creating model object: ' + fname)

    # load brain object
    bo = se.load(sys.argv[1])

    # filter
    bo.get_filtered_bo()

    npz_outfile = sys.argv[1][:-2]+'npz'
    if not os.path.isfile(npz_outfile):
        np.savez(npz_outfile, Y = bo.get_data().as_matrix(),
                          R = bo.get_locs().as_matrix(),
                          fname_labels=np.atleast_2d(bo.sessions.as_matrix()),
                          sample_rate=bo.sample_rate)


    # exapand matrix
    num_corrmat_x, denom_corrmat_x, n_subs = _bo2model(bo, R, int(radius))

    #### save the z expanded correlation matrix
    print('saving model object: ' + fname)
    np.savez(os.path.join(results_dir, fname), C_est=np.divide(num_corrmat_x, denom_corrmat_x))

else:
    print('skipping model (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])