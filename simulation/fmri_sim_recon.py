
import supereeg as se
from supereeg.helpers import _corr_column, get_rows, known_unknown, remove_electrode
import numpy as np
import sys
import os
from config import config

bos_fname = sys.argv[1]

file_name = os.path.basename(os.path.splitext(bos_fname)[0])

bo_dir = config['bof_datadir']

results_dir = os.path.join(config['recon_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

mo_mo_fname = os.path.join(config['model_datadir'], file_name + '.mo')
mo_mo = se.load(mo_mo_fname)

ave_dir = config['ave_datadir']

ave ='sub_locs_ave_model.mo'
# ave = 'ave_model.mo'

mo = se.load(os.path.join(ave_dir, ave))

## if we want to add pyfr info into it:

# pyfr_ave ='pyfr_ave_mat.mo'
# mo_pyfr = se.load(os.path.join(ave_dir, pyfr_ave))
#
# new_mo = se.Model(mo_pyfr, locs = mo.get_locs(), n_subs = mo_pyfr.n_subs)
#
# mo = new_mo + mo

## if we want to remove subjects data:
#mo = mo - mo_mo

R = mo.get_locs().values

bo_s = se.load(bos_fname)
bo_s.filter=None

bo_fname = os.path.join(bo_dir, 'sub-' + file_name.split('_')[-1] + '.bo')
bo = se.load(bo_fname)
bo.filter=None


R_subj = bo.get_locs().values
R_sub_subj = bo_s.get_locs().values


known_inds, unknown_inds = known_unknown(R_subj, R, R)

actual = bo[:,known_inds]

print('bo indexed')

recon_outfile = os.path.join(results_dir, os.path.basename(file_name + '.npz'))

bo_r = mo.predict(bo_s)

R_actual = actual.get_locs().values

R_recon = bo_r.get_locs().values

known_inds, unknown_inds = known_unknown(R_recon, R_actual)

bo_r = bo_r[:,known_inds]

c = _corr_column(bo_r.data.as_matrix(), actual.get_zscore_data())

print(c)

np.savez(recon_outfile, corrs=c, locs=bo_r.get_locs().values)