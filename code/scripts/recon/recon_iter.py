import supereeg as se
from supereeg.helpers import _corr_column, get_rows, known_unknown, remove_electrode
import numpy as np
import sys, time
import os
from config import config
from bandbrain import BandBrain
import warnings
import gc

warnings.simplefilter('ignore')

bo_fname = sys.argv[1]

freq = bo_fname.split('_')[-1].split('.bo')[0]

if not freq in set(['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma', 'broadband', 'raw']):
    freq = 'raw'

results_dir = os.path.join(config['resultsdir'], freq + '_recon')

file_name = os.path.basename(os.path.splitext(bo_fname)[0])
print(file_name)

mo_mo_fname = os.path.join(config['modeldir'], file_name + '.mo')
numtries = 0
loaded = False
while not loaded and numtries < 20:
    try:
        mo_mo = se.load(mo_mo_fname)
        loaded = True
    except:
        numtries += 1
        time.sleep(5)

ave_dir = os.path.join(config['resultsdir'])

ave ='ave_mat_' + freq + '.mo'

numtries = 0
loaded = False
while not loaded and numtries < 20:
    try:
        mo = se.load(os.path.join(ave_dir, ave))
        loaded = True
    except:
        numtries += 1
        time.sleep(5)

R = mo.get_locs().values

mo_s = mo - mo_mo

del mo
del mo_mo
gc.collect()

print('models deleted')

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

freq_bo = se.load(bo_fname)
basefname = os.path.basename(bo_fname)
freq = bo_fname.split('_')[-1].split('.bo')[0]
og_fname = os.path.join(config['og_bodir'], basefname.split('_' + freq)[0] + '.bo')
if not freq in set(['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma', 'broadband', 'raw']):
    og_fname = bo_fname
og_bo = se.load(og_fname)
filter_inds = og_bo.filter_inds
kurtosis = og_bo.kurtosis
bo = BandBrain(freq_bo, og_bo)
num_elec = bo.get_locs().shape[0]
del bo
del freq_bo
gc.collect()

print('filter applied')

def recon_elec(elec_ind):
    recon_outfile_across = os.path.join(results_dir, os.path.basename(os.path.splitext(bo_fname)[0] + '_' + str(elec_ind) + '.npz'))
    recon_outfile_within = os.path.join(results_dir, os.path.basename(os.path.splitext(bo_fname)[0] + '_' + str(elec_ind) + '_within.npz'))
    if os.path.exists(recon_outfile_across) and os.path.exists(recon_outfile_within):
        return
    
    freq_bo = se.load(bo_fname)
    bo = BandBrain(freq_bo, og_bo)
    electrode = bo.get_locs().iloc[elec_ind]

    R_K_subj = bo.get_locs().values

    R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)

    known_inds, unknown_inds, e_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)

    electrode_ind = get_rows(R, electrode.values)
    actual = bo[:,elec_ind]
    bo = bo[:, other_inds]

    print('bo indexed for ' + str(elec_ind))
    
    if not os.path.exists(recon_outfile_across):
        bo_r = mo_s.predict(bo, recon_loc_inds=e_ind)

        print(electrode)
        c = _corr_column(bo_r.data.as_matrix(), actual.get_zscore_data())

        print(c)

        np.savez(recon_outfile_across, coord=electrode, corrs=c)

    if not os.path.exists(recon_outfile_within):

        Model = se.Model(bo, locs=R_K_subj)

        m_locs = Model.get_locs().as_matrix()
        known_inds, unknown_inds, e_ind = known_unknown(m_locs, R_K_removed, m_locs, elec_ind)
        bo_r = Model.predict(bo)

        bo_r = bo_r[:, unknown_inds]

        print(electrode)

        c = _corr_column(bo_r.data.as_matrix(), actual.get_zscore_data())

        print(c)

        np.savez(recon_outfile_within, coord=electrode, corrs=c)

for elec_ind in range(num_elec):
    recon_elec(elec_ind)
    gc.collect()
