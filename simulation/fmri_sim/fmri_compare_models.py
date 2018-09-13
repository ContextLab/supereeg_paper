import supereeg as se
import numpy as np
import sys
import os
from supereeg.helpers import get_rows
from supereeg.helpers import _z2r, _r2z
import glob as glob
from scipy.spatial.distance import cdist
from config import config


results_dir = os.path.join(config['compare_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

fname = sys.argv[1]

model_dir = os.path.join(config['model_datadir'])

bo_dir = config['bof_datadir']

fname = os.path.basename(os.path.splitext(fname)[0])

print('creating model object: ' + fname)

# load brain object
path_info, i, j = sys.argv[1][:-3].rsplit("_", 2)

bo = se.load(os.path.join(bo_dir, 'sub-%d.bo' % int(j)))

# filter
bo.filter=None

mo_fname = (sys.argv[1][:-3]+ '.mo').replace('bo_sub', 'models')

# make model
mo = se.load(mo_fname)

ave_dir = config['ave_datadir']

ave ='sub_locs_ave_model.mo'

ave_mo = se.load(os.path.join(ave_dir, ave))

sub_inds = get_rows(bo.get_locs().values, ave_mo.get_locs().values)

subbo = bo[:, sub_inds]

true_model = se.Model(subbo, locs = ave_mo.get_locs().values)

d_mo = (1. - cdist(mo.get_model(), true_model.get_model(), metric='correlation'))

c_d_mo = np.diagonal(d_mo).mean()

d_ave = (1. - cdist(ave_mo.get_model(), true_model.get_model(), metric='correlation'))

c_d_ave = np.diagonal(d_ave).mean()


recon_outfile = os.path.join(results_dir, os.path.basename(fname + '.npz'))

np.savez(recon_outfile, c_d_ave=c_d_ave, c_d_mo=c_d_mo)