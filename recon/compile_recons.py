
from stats import compile_corrs, density
from config import config
import os
import glob as glob
import pandas as pd
import sys


model_template = sys.argv[1]

radius = sys.argv[2]

results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)

dir = os.path.join(results_dir, sys.argv[3])


files = glob.glob(os.path.join(dir, '*.npz'))

all_corrs = pd.DataFrame()

for i in files:

    compile_temp = compile_corrs(config['datadir'], i)
    if all_corrs.empty:
        all_corrs = compile_temp
    else:
        all_corrs = all_corrs.append(compile_temp)
        all_corrs.to_csv(os.path.join(dir, 'all_corrs.csv'))

all_corrs['Density'] = density(all_corrs['R'].tolist(), 3)

print(os.path.basename(dir) + ': ' + all_corrs['Correlation'].mean())
all_corrs.to_csv(os.path.join(dir, 'all_corrs.csv'))
