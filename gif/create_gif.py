import supereeg as se
from supereeg.helpers import make_gif_pngs
import sys
import os
from config import config



model_template = sys.argv[1]

if model_template == 'pyFR_union':
    model = se.load(os.path.join(config['pyFR_union_dir'],'pyFR_union.mo'))

elif model_template == 'gray_mask_6mm_brain':
    model = se.load(os.path.join(config['gray_mask_6mm_brain_dir'], 'gray_mask_6mm_brain.mo'))

else:
    model = se.load(intern(model_template))

results_dir = os.path.join(config['resultsdir'], model_template)


try:
    if not os.path.exists(os.path.dirname(results_dir)):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

bo = se.load('example_data')
bo.info()


bor = model.predict(bo)

nii = bor.to_nii()

make_gif_pngs(nii, gif_path=results_dir, display_mode='lyrz', threshold=0, plot_abs=False, colorbar='False',
                            vmin=-20, vmax=20,)


