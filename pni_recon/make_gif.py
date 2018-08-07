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
            'colorbar': True,
            'vmin': -.25,
            'vmax': .25}

fname = 'nn_brain_data_5000_'+ model_template+'_'+vox_size+'.bo'


bo = se.load(os.path.join(results_dir, fname))

control = np.load(os.path.join(results_dir,'audio_control_5000.npy'))

audio = np.load(os.path.join(results_dir,'audio_5000.npy'))

pd_audio = pd.DataFrame(audio)
pd_control = pd.DataFrame(control)

rolling =bo.get_data().rolling(window=int(window)).corr(other=pd_audio[0])

bo_s = se.Brain(data=rolling[int(window):], locs = bo.get_locs(), sample_rate=512)

rolling_c =bo.get_data().rolling(window=int(window)).corr(other=pd_control[0])

bo_c = se.Brain(data=rolling_c[int(window):], locs = bo.get_locs(), sample_rate=512)

bo_s.save(os.path.join(results_dir,'audio_'+ window + '_' + fname))
bo_c.save(os.path.join(results_dir,'control_'+ window + '_' +fname))

bo_nii = bo_s.to_nii(vox_size=int(vox_size))
bo_nii_c = bo_c.to_nii(vox_size=int(vox_size))

gif_audio_dir = os.path.join(results_dir, 'gif_audio' + window)
gif_control_dir = os.path.join(results_dir, 'gif_control' + window)

try:
    if not os.path.exists(gif_audio_dir):
        os.makedirs(gif_audio_dir)
except OSError as err:
   print(err)


try:
    if not os.path.exists(gif_control_dir):
        os.makedirs(gif_control_dir)
except OSError as err:
   print(err)

bo_nii.make_gif(gif_audio_dir,name='audio', index=range(100, 500, 1), **gif_args)

bo_nii_c.make_gif(gif_control_dir,name='control', index=range(100, 500, 1), **gif_args)