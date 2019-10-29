
from config import config
import os
import glob as glob
import pandas as pd
import sys
from sklearn.neighbors import NearestNeighbors
import numpy as np
import supereeg as se

### for cluster:
model_template = sys.argv[1]

radius = sys.argv[2]

dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)

#dir = os.path.join(results_dir, sys.argv[3])


### locally
#dir = sys.argv[1]



def z2r(z):
    """
    Function that calculates the inverse Fisher z-transformation

    Parameters
    ----------
    z : int or ndarray
        Fishers z transformed correlation value

    Returns
    ----------
    result : int or ndarray
        Correlation value


    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def r2z(r):
    """
    Function that calculates the Fisher z-transformation

    Parameters
    ----------
    r : int or ndarray
        Correlation value

    Returns
    ----------
    result : int or ndarray
        Fishers z transformed correlation value


    """
    return 0.5 * (np.log(1 + r) - np.log(1 - r))

def compile_corrs(bo_path, corr_path, threshold=10):
    """
        Compiles correlation values - as well as other subject/electrode specific paramters - creates the compiled pandas dataframe used for figures

        Parameters
        ----------
        path_to_npz_data : string
            Path to npz files - I know this isn't a great way to do this :/

        corr_path : npz file
            npz file containing correlation values (loop outside - for each electrode)

        Returns
        ----------
        results : dataframe
            compiled dataframe with: Subject, electrode, correlation, samples, and sample rate

        """
    def parse_path_name(path_name):
        if os.path.basename(path_name).count('_') == 1:
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_",1)[0]
            electrode = os.path.splitext(os.path.basename(path_name))[0].split("_",1)[1]
            return f_name, electrode
        elif os.path.basename(path_name).count('_') == 2:
            f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_",2)[0:2])
            electrode = os.path.splitext(os.path.basename(path_name))[0].split("_",2)[2]
            return f_name, electrode
        else:
            return "error"
    ### parse path is necessary for the wacky naming system
    f_name, electrode = parse_path_name(corr_path)
    corr_data = np.load(corr_path, mmap_mode='r')
    tempR = np.round(corr_data['coord'], 2)
    tempmeancorr = z2r(np.mean(r2z(corr_data['corrs'])))
    tempsamplerate = np.mean(se.load(os.path.join(bo_path, f_name + '.bo'), field='sample_rate'))
    tempsamples = se.load(os.path.join(bo_path, f_name + '.bo'), field='sessions').shape[0]
    kurt_vals = se.load(os.path.join(bo_path, f_name + '.bo'), field='kurtosis')
    thresh_bool = kurt_vals > threshold
    tempelecs = sum(~thresh_bool)
    tempsessions = se.load(os.path.join(bo_path, f_name + '.bo'), field='sessions').max()
    tempthresholded = sum(thresh_bool)

    return pd.DataFrame({'R': [tempR], 'Correlation': [tempmeancorr], 'Subject': [f_name], 'Electrode': [electrode],
                         'Sample rate' : [tempsamplerate], 'Samples': [tempsamples], 'Total Electrodes': [tempelecs],
                         'Sessions': [tempsessions], 'Number thresholded': [tempthresholded]})


def density(n_by_3_Locs, nearest_n):
    """
        Calculates the density of the nearest n neighbors

        Parameters
        ----------

        n_by_3_Locs : ndarray
            Array of electrode locations - one for each row

        nearest_n : int
            Number of nearest neighbors to consider in density calculation

        Returns
        ----------
        results : ndarray
            Denisity for each electrode location

        """
    nbrs = NearestNeighbors(n_neighbors=nearest_n, algorithm='ball_tree').fit(n_by_3_Locs)
    distances, indices = nbrs.kneighbors(n_by_3_Locs)
    return np.exp(-(distances.sum(axis=1) / (np.shape(distances)[1] - 1)))




def compile_corrs_new(corr_path):
    """
        Compiles correlation values - as well as other subject/electrode specific paramters - creates the compiled pandas dataframe used for figures

        Parameters
        ----------
        path_to_npz_data : string
            Path to npz files - I know this isn't a great way to do this :/

        corr_path : npz file
            npz file containing correlation values (loop outside - for each electrode)

        Returns
        ----------
        results : dataframe
            compiled dataframe with: Subject, electrode, correlation, samples, and sample rate

        """
    def parse_path_name(path_name):

        underscore_count = os.path.basename(path_name).count('_')
        if os.path.splitext(os.path.basename(path_name))[0].split("_", underscore_count)[-1] == 'within':
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", underscore_count)[0]
            electrode = os.path.splitext(os.path.basename(path_name))[0].split("_", underscore_count)[-2]
        else:
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", underscore_count)[0]
            electrode = os.path.splitext(os.path.basename(path_name))[0].split("_", underscore_count)[-1]


        return f_name, electrode

    ### parse path is necessary for the wacky naming system
    f_name, electrode = parse_path_name(corr_path)
    corr_data = np.load(corr_path, mmap_mode='r')
    tempR = np.round(corr_data['coord'], 2)
    tempmeancorr = z2r(np.mean(r2z(corr_data['corrs'])))


    return pd.DataFrame({'R': [tempR], 'Correlation': [tempmeancorr], 'Subject': [f_name], 'Electrode': [electrode]})



#freqnames = ['raw', 'delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma', 'broadband']

freqnames = ['raw']

for r in ['ram', 'pyfr']:

    for freq in freqnames:

        within_files = glob.glob(os.path.join('/dartfs/rc/lab/D/DBIC/CDL/f003f64', r+ '_results', freq+'_recon', '*_within.npz'))
        across_files= list(set(glob.glob(os.path.join('/dartfs/rc/lab/D/DBIC/CDL/f003f64', r+ '_results', freq+'_recon', '*.npz'))) -
                           set(glob.glob(os.path.join('/dartfs/rc/lab/D/DBIC/CDL/f003f64', r+ '_results', freq+'_recon', '*_within.npz'))))
        all_corrs_across = pd.DataFrame()
        for i in across_files:

            compile_temp = compile_corrs_new(i)
            if all_corrs_across.empty:
                all_corrs_across = compile_temp
            else:
                all_corrs_across = all_corrs_across.append(compile_temp)
        all_corrs_across.to_csv(os.path.join('/dartfs/rc/lab/D/DBIC/CDL/f002s72/freq_plots', r+ '_results', freq + '_across.csv'))
        all_corrs_within = pd.DataFrame()
        for i in within_files:
            compile_temp = compile_corrs_new(i)
            if all_corrs_within.empty:
                all_corrs_within = compile_temp
            else:
                all_corrs_within = all_corrs_within.append(compile_temp)
        all_corrs_within.to_csv(os.path.join('/dartfs/rc/lab/D/DBIC/CDL/f002s72/freq_plots', r+ '_results', freq + '_within.csv'))


