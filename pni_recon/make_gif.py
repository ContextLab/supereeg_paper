import supereeg as se
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from config import config
plt.switch_backend('agg')


### for cluster:
model_template = sys.argv[1]

vox_size = sys.argv[2]

results_dir = os.path.join(config['resultsdir'], model_template+ '_' + vox_size)

window = sys.argv[3]

cmap=plt.cm.get_cmap('RdBu')

gif_args = {'cmap': cmap,
            'display_mode': 'lyrz',
            'threshold': 0,
            'plot_abs': False,
            'colorbar': False,
            'vmin': -.1,
            'vmax': .1}


bo = se.load(os.path.join(results_dir, 'nn_brain_data_5000_gray_20.bo'))

audio = np.load(os.path.join(results_dir,'audio_5000.npy'))

pd_audio = pd.DataFrame(audio)

rolling =bo.get_data().rolling(window=int(window)).corr(other=pd_audio[0])

bo_s = se.Brain(data=rolling[int(window):], locs = bo.get_locs(), sample_rate=512)

bo_nii = bo_s.to_nii(vox_size=20)

gif_dir = os.path.join(results_dir, 'gif_' + window)

try:
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
except OSError as err:
   print(err)

bo_nii.make_gif(gif_dir,name='recon', index=np.arange(20), **gif_args)