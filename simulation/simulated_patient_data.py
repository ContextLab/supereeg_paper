# -*- coding: utf-8 -*-
"""
=============================
Simulate data
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import supereeg as se
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from supereeg.helpers import _corr_column
from config import config

try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])

model = se.load('example_model')

locs = model.locs

# n_electrodes - number of electrodes for reconstructed patient
n_elecs = range(10, 100, 10)


# m_patients - number of patients in the model
m_patients = [5, 10]


# m_electrodes - number of electrodes for each patient in the model
m_elecs = range(10, 100, 10)


iter_val = 5

append_d = pd.DataFrame()

param_grid = [(p, m, n) for p in m_patients for m in m_elecs for n in n_elecs]

for p, m, n in param_grid:
    d = []

    for i in range(iter_val):
        # create brain objects with m_patients and loop over the number of model locations and subset locations to build model
        model_bos = [se.simulate_model_bos(n_samples=100, sample_rate=1000, locs=locs, sample_locs=m, noise =.3) for x in range(p)]

        # create model from subsampled gray locations
        model = se.Model(model_bos, locs=locs)

        # brain object locations subsetted entirely from both model and gray locations
        sub_locs = locs.sample(n).sort_values(['x', 'y', 'z'])

        # simulate brain object
        bo = se.simulate_bo(n_samples=100, sample_rate=1000, locs=locs, noise =.3)

        # parse brain object to create synthetic patient data
        data = bo.data.iloc[:, sub_locs.index]

        # create synthetic patient (will compare remaining activations to predictions)
        bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)

        # reconstruct at 'unknown' locations
        bo_r = model.predict(bo_sample)

        # find the reconstructed indices
        recon_inds = [i for i, x in enumerate(bo_r.label) if x == 'reconstructed']

        # sample reconstructed data a reconstructed indices
        recon = bo_r.data.iloc[:, recon_inds]

        # sample actual data at reconstructed locations
        actual = bo.data.iloc[:, recon_inds]

        # correlate reconstruction with actual data
        corr_vals = _corr_column(actual.as_matrix(), recon.as_matrix())
        corr_vals_sample = np.random.choice(corr_vals, 5)

        d.append(
            {'Subjects in model': p, 'Electrodes per subject in model': m, 'Electrodes per reconstructed subject': n,
             'Average Correlation': corr_vals_sample.mean(), 'Correlations': corr_vals})

    d = pd.DataFrame(d, columns=['Subjects in model', 'Electrodes per subject in model',
                                 'Electrodes per reconstructed subject', 'Average Correlation', 'Correlations'])
    append_d = append_d.append(d)
    append_d.index.rename('Iteration', inplace=True)

new_df = append_d.groupby('Average Correlation').mean()

#fig, axs = plt.subplots(ncols=len(np.unique(new_df['Subjects in model'])), sharex=True, sharey=True)
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)

axs_iter = 0

cbar_ax = fig.add_axes([.92, .3, .03, .4])

fig.subplots_adjust(right=0.85)
fig.set_size_inches(14,5)
for i in np.unique(new_df['Subjects in model']):
    data_plot = append_d[append_d['Subjects in model'] == i].pivot_table(index=['Electrodes per subject in model'],
                                                                         columns='Electrodes per reconstructed subject',
                                                                         values='Average Correlation')
    axs[axs_iter].set_title('Patients = ' + str(i))
    sns.heatmap(data_plot, cmap="coolwarm", cbar=axs_iter == 0, ax=axs[axs_iter], cbar_ax=None if axs_iter else cbar_ax)
    axs[axs_iter].invert_yaxis()
    axs_iter += 1


plt.savefig(os.path.join(config['resultsdir'], str(sys.argv[1]) + '_heatmap.pdf'))


