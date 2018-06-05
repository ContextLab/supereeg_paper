
from stats import compile_corrs, density
from config import config
import os
import glob as glob
import pandas as pd
import sys


model_template = sys.argv[1]

radius = sys.argv[2]

results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)

across_dir = os.path.join(results_dir, 'across_subjects')
within_dir = os.path.join(results_dir, 'within_subjects')
all_dir = os.path.join(results_dir, 'all_subjects')

dirs = [across_dir, within_dir, all_dir]

for d in dirs:

    files = glob.glob(os.path.join(d, '*.npz'))

    all_corrs = pd.DataFrame()

    for i in files:

        compile_temp = compile_corrs(config['datadir'], i)
        if all_corrs.empty:
            all_corrs = compile_temp
        else:
            all_corrs = all_corrs.append(compile_temp)

    all_corrs['Density'] = density(all_corrs['R'].tolist(), 3)

    print(os.path.basename(d) + ': ' + all_corrs['Correlation'].mean())
    all_corrs.to_csv(os.path.join(d, 'all_corrs.csv'))
