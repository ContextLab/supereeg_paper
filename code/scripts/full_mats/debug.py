
import supereeg as se
import numpy as np
import sys
import os
from config import config
from bandbrain import BandBrain

sys.path.append(os.path.dirname(__file__))


fname = os.path.join(os.path.dirname(__file__), 'R1065J_RAM_FR1_0_broadband.bo')

def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)

locs_file =  os.path.join(os.path.dirname(__file__), 'broadband_locs.npz')
R = np.load(locs_file)['locs']

elec_count = electrode_search(os.path.join(os.path.dirname(__file__), 'R1065J_RAM_FR1_0.bo'))


if elec_count > 1:

    print('creating model object: ' + fname)

    # load brain object
    bo = se.load(fname)

    # turn it into my fancy ~BandBrain~
    bo = BandBrain(bo)

    # load original brain object
    og_bo = se.load(os.path.join(os.path.dirname(__file__), 'R1065J_RAM_FR1_0.bo'))
    # og_bo.apply_filter()
    # mo = se.Model(og_bo, locs=R)

    # filter
    bo.apply_filter(og_bo)

    # turn it back into a vanilla Brain
    bo = se.Brain(bo)

    # make model
    mo = se.Model(bo, locs=R)

    # save model
    mo.save('out.mo')

else:
    print('skipping model (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])