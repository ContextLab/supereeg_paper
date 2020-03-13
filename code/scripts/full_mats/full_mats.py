import supereeg as se
import numpy as np
import sys
import os
import time
from config import config
from bandbrain import BandBrain


fname = sys.argv[1]
results_dir = config['resultsdir']
freq = fname.split('_')[-1].split('.bo')[0]
fname = os.path.basename(os.path.splitext(fname)[0])

if os.path.exists(os.path.join(results_dir, fname)):
    print('file already exists')
    exit()


def electrode_search(fname, threshold=10):
    # searching original .bo
    fname = os.path.basename(os.path.splitext(fname)[0])
    fname = os.path.join(config['og_bodir'], fname.split('_' + freq)[0] + '.bo')
    try:
        kurt_vals = se.load(fname, field='kurtosis')
    except:
        kurt_vals = se.load(sys.argv[1], field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)

try:
    locs_file = os.path.join(config['locsdir'], freq + '_locs.npz')
    R = np.load(locs_file)['locs']
except:
    locs_file = os.path.join(config['locsdir'], 'raw_locs.npz')
    R = np.load(locs_file)['locs']

elec_count = electrode_search(sys.argv[1])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


if elec_count > 1: 

    print('creating model object: ' + fname)

    # load brain object
    numtries = 0
    loaded = False
    while numtries < 20 and not loaded:
        try:
            bo = se.load(sys.argv[1])
            loaded = True
        except:
            numtries += 1
            time.sleep(5)
    bo = se.load(sys.argv[1])

    # load original brain object
    og_fname = os.path.join(config['og_bodir'], fname.split('_' + freq)[0] + '.bo')
    try:
        og_bo = se.load(og_fname)
    except:
        og_bo = se.load(sys.argv[1])
    og_bo.update_filter_inds()

    # turn it into fancy ~BandBrain~
    bo = BandBrain(bo, og_bo=og_bo, og_filter_inds=og_bo.filter_inds)

    # filter
    bo.apply_filter()

    # turn it back into a vanilla Brain
    bo = se.Brain(bo)

    # make model
    mo = se.Model(bo, locs=R)

    # save model
    mo.save(os.path.join(results_dir, fname))

else:
    print('skipping model (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])
