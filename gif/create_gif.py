import supereeg as se
from supereeg.helpers import make_gif_pngs
import sys
import os
import copy
import pandas as pd
from config import config
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')


model_template = sys.argv[1]

model_dir = os.path.join(config['datadir'], model_template)

results_dir = os.path.join(config['resultsdir'], model_template)
print(results_dir)

try:
    os.stat(results_dir)
except:
    os.makedirs(results_dir)

model_data = os.path.join(model_dir, model_template + '.mo')

print(model_data)

model = se.load(intern(model_data))

bo = se.load('example_data')
bo.info()

bor = model.predict(bo)

zbo = copy.copy(bor)
zbo.data = pd.DataFrame(bor.get_zscore_data())

if model_template == 'gray_mask_6mm_brain':
    nii = zbo.to_nii(template='6mm')
else:
    nii = zbo.to_nii(template='20mm')

make_gif_pngs(nii, gif_path=results_dir, display_mode='lyrz', threshold=0, plot_abs=False, colorbar=False,
                            vmin=-5, vmax=5,)


