
from config import config
import os
import glob as glob
import pandas as pd
import sys
from sklearn.neighbors import NearestNeighbors
import numpy as np
import supereeg as se

### for cluster:

data = sys.argv[1]

chunk_size = 10000

rand_iters = 2

results_dir = os.path.join(config['resultsdir'], data)


def compile_corrs(file_path):
    """
        Compiles correlation values - as well as other subject specific paramters - creates the compiled pandas dataframe used for figures

        Parameters
        ----------

        file_path : string
            path to npz file containing correlation values

        Returns
        ----------
        results : dataframe
            compiled dataframe with: Subject, electrode, correlation, samples, and sample rate

        """

    data = np.load(file_path, mmap_mode='r')
    f_name = os.path.splitext(os.path.basename(file_path))[0]

    return pd.DataFrame({'rand': data['rand'], 'apart': data['apart'], 'close': data['close'],
                         'Subject': f_name, 'sample_rate': data['sample_rate'], 'samples': data['samples'],
                         'chunk_size': data['chunk_size']})


files = glob.glob(os.path.join(results_dir, '*.npz'))

all_corrs = pd.DataFrame()

for f in files:

    compile_temp = compile_corrs(f)
    if all_corrs.empty:
        all_corrs = compile_temp
    else:
        all_corrs = all_corrs.append(compile_temp)


print('Nans for :' + str(all_corrs['Subject'][all_corrs['rand'].isna()].unique()))

all_corrs.dropna(inplace=True)

all_corrs.to_csv(os.path.join(config['resultsdir'], data + '_time_stability.csv'))

