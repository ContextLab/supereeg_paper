
import supereeg as se
from supereeg.helpers import _corr_column, get_rows, known_unknown, remove_electrode
import numpy as np
import sys
import os
from config import config

bo_fname = sys.argv[1]

file_name = os.path.basename(os.path.splitext(bo_fname)[0])

elec_ind = int(sys.argv[2])

model_template = sys.argv[3]

radius = sys.argv[4]

mo_mo_fname = os.path.join(config['modeldir'], model_template + '_' + radius, file_name + '.mo')
mo_mo = se.load(mo_mo_fname)

ave_dir = os.path.join(config['avedir'], model_template+ '_' + radius)

ave ='ave_mat.mo'
mo = se.load(os.path.join(ave_dir, ave))

R = mo.get_locs().as_matrix()

mo_s = mo - mo_mo

mo = None
mo_mo = None

print('models deleted')
results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

bo = se.load(bo_fname)
bo.apply_filter()

print('filter applied')

electrode = bo.get_locs().iloc[elec_ind]

R_K_subj = bo.get_locs().as_matrix()

R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)

known_inds, unknown_inds, e_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)

electrode_ind = get_rows(R, electrode.values)
actual = bo[:,elec_ind]
bo = bo[:, other_inds]

print('bo indexed')

recon_outfile_across = os.path.join(results_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))

recon_outfile_within = os.path.join(results_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '_within.npz'))

if not os.path.exists(recon_outfile_across):
    bo_r = mo_s.predict(bo, recon_loc_inds=e_ind)


    print(bo_r.get_locs())
    print(electrode)
    c = _corr_column(bo_r.data.as_matrix(), actual.get_zscore_data())

    print(c)

    np.savez(recon_outfile_across, coord=electrode, corrs=c)

elif not os.path.exists(recon_outfile_within):

    Model = se.Model(bo, locs=R_K_subj)

    m_locs = Model.get_locs().as_matrix()
    known_inds, unknown_inds, e_ind = known_unknown(m_locs, R_K_removed, m_locs, elec_ind)
    bo_r = Model.predict(bo)

    bo_r = bo_r[:, unknown_inds]

    print(bo_r.get_locs())
    print(electrode)

    c = _corr_column(bo_r.data.as_matrix(), actual.get_zscore_data())

    print(c)

    np.savez(recon_outfile_within, coord=electrode, corrs=c)

else:
    print('both within and across reconstructions are done')