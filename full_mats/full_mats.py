
import supereeg as se
import numpy as np
import sys
import os
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

radius = sys.argv[3]

vox_size = sys.argv[4]

results_dir = os.path.join(config['resultsdir'], model_template +"_"+ vox_size)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)

if model_template == 'pyFR_union':
    locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')

    R = np.load(locs_file)['locs']
else:

    R_nii = se.load(model_template, vox_size=int(vox_size))
    R = R_nii.get_locs().as_matrix()

elec_count = electrode_search(sys.argv[1])

if elec_count > 1:

    fname =os.path.basename(os.path.splitext(fname)[0])

    print('creating model object: ' + fname)

    # load brain object
    bo = se.load(sys.argv[1])

    # filter
    bo.apply_filter()

    # make model
    mo = se.Model(bo, locs=R)

    # save model
    mo.save(os.path.join(results_dir, fname))

else:
    print('skipping model (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])