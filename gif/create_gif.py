import supereeg as se
from supereeg.helpers import make_gif_pngs
import sys
import os
import copy
import pandas as pd
from config import config
import matplotlib.pyplot as plt
plt.switch_backend('agg')


model_template = sys.argv[1]

model_dir = os.path.join(config['datadir'], model_template)

results_recon_dir = os.path.join(config['resultsdir'], model_template + '_recon')

results_obs_dir = os.path.join(config['resultsdir'], model_template + '_obs')

try:
    os.stat(results_recon_dir)
except:
    os.makedirs(results_recon_dir)

try:
    os.stat(results_obs_dir)
except:
    os.makedirs(results_obs_dir)

model_data = os.path.join(model_dir, model_template + '.mo')


model = se.load(intern(model_data))

bo = se.load('example_data')
bo.info()

bor = model.predict(bo)

zbor = copy.copy(bor)
zbor.data = pd.DataFrame(bor.get_zscore_data())

zbo = copy.copy(bo)
zbo.data = pd.DataFrame(bo.get_zscore_data())

if model_template == 'gray_mask_6mm_brain':
    nii_recon = zbor.to_nii(template='6mm')
    nii_obs = zbo.to_nii(template='6mm')
else:
    nii_recon = zbor.to_nii(template='20mm')
    nii_obs = zbor.to_nii(template='20mm')

make_gif_pngs(nii_recon, gif_path=results_recon_dir, window_min=1000, window_max=1256, display_mode='lyrz', threshold=0,
              plot_abs=False, colorbar=False, vmin=-5, vmax=5)


make_gif_pngs(nii_obs, gif_path=results_obs_dir, window_min=1000, window_max=1256, display_mode='lyrz', threshold=0,
              plot_abs=False, colorbar=False, vmin=-5, vmax=5)
