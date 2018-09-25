import os
import supereeg as se
import hypertools as hyp
import pandas as pd
import numpy as np
import glob as glob
from scipy import stats, signal
import numpy.matlib as mat
import matplotlib.patches as patches
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors
from nilearn import plotting as ni_plt
from supereeg.helpers import _log_rbf, _brain_to_nifti, _plot_borderless
from supereeg.helpers import _corr_column, get_rows, known_unknown, remove_electrode
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats


def plot_hist(dataframe, title=None, outfile=None):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    sns.set_style("white")
    df_corrs = pd.DataFrame()
    df_corrs['Correlations'] = dataframe['Correlation'].values
    n_count = len(df_corrs)
    bin_values = np.arange(start=-1, stop=1, step=.025)
    ax = sns.distplot(df_corrs, hist=True, kde=True,
             bins=bin_values, color = 'k',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

    vals = ax.get_yticks()
    ax.set_yticklabels([np.round(x/n_count,5) for x in vals])
    ax.set_ylabel('Proportion of electrodes', fontsize=21)
    ax.set_xlabel('Correlation', fontsize=21)
    ax.set_xlim(-1, 1)
    ax.set_frame_on(False)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    #plt.text(2,10, 'mean = '+ str(np.round(dataframe['Correlation'].mean(),3)))
    left, width = .05, .5
    bottom, height = .75, .5
    plt.tight_layout()

    ax.text(left, bottom, 'mean = '+ str(np.round(dataframe['Correlation'].mean(),3)),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=18)

    if outfile:
        plt.savefig(outfile)
        plt.clf()

def plot_2_histograms(df, X, Y, xticks=True, legend=True, outfile=None):

    grouped_results = df.groupby('Subject')[Y, X].mean()
    t_stat_group = stats.ttest_rel(grouped_results[Y],grouped_results[X])

    fig = plt.gcf()
    fig.set_size_inches(18.5, 8.5)
    bin_values = np.arange(start=-1, stop=1, step=.01)

    ax = sns.distplot(z2r(df[X]), hist=True, kde=True,
             bins=bin_values, color = 'k',
             hist_kws={'edgecolor':'lightgray', 'alpha':.3},
             kde_kws={'linewidth': 4, 'alpha':1, 'color':'lightgray'})


    ax = sns.distplot(z2r(df[Y]), hist=True, kde=True,
             bins=bin_values, color = 'k',
             hist_kws={'edgecolor':'k', 'alpha':.7},
             kde_kws={'linewidth': 4, 'alpha':1, 'color':'k'})

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    left, width = .05, .5
    bottom, height = .25, .5
    n_count = len(df[Y].values)
    vals = ax.get_yticks()

    ax.set_yticklabels([np.round(x/n_count,5) for x in vals])
    ax.set_xlim(-1, 1)

    if legend:
        leg = ax.legend(['Within', 'Across'], fontsize=50, loc='upper left')
        LH = leg.legendHandles
        LH[1].set_color('k')
        LH[1].set_alpha(1)

    ylim = ax.get_ylim()[1]
    # ax.plot(z2r(df[X]).mean(), ylim, 'vline', color='k')
    # ax.plot(z2r(df[Y]).mean(), ylim, 'vline', color='k')
    #plt.plot(z2r(df[X]).mean(), ylim, z2r(df[Y]).mean(), ylim, marker = '|', color='k')
    m1, n1 = [z2r(df[X]).mean(), z2r(df[Y]).mean()], [ylim, ylim]
    plt.plot(m1, n1, marker = '|', mew=4, markersize=20, color='k', linewidth=4)
    a1 = (z2r(df[X]).mean() + z2r(df[Y]).mean()) /2
    print(a1)
    ax.plot(a1, ylim + .1, marker = '*', markersize=20, color='k')
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel('Proportion \n of electrodes', fontsize=50)

    if not xticks:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Correlation', fontsize=50)
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 2 == 0:
                label.set_visible(False)

    plt.tight_layout()


    if outfile:
        plt.savefig(outfile)

    print(t_stat_group)


def plot_column(X, Y, title=None, outfile=None):
    mpl.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots()
    ax.scatter(X, Y, color='k', alpha=.1)
    #ax.set_xscale('log')
    ax.set_title(title)
    ax.set_ylabel(X.name)
    ax.set_xlabel(Y.name)
    left, width = .05, .5
    bottom, height = .05, .5
    rstat = stats.pearsonr(X, Y)
    ax.text(left, bottom, 'r = ' + str(np.round(rstat[0],2)) + ' p = ' + str(rstat[1]),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)

def plot_times_series(time_data_df, lower_bound, upper_bound, outfile = None):
    """
    Plot reconstructed timeseries
    Parameters
    ----------
    time_data_df : pandas dataframe
        Dataframe with reconstructed and actual timeseries
    lower_bound : int
        Lower bound for timeseries
    upper_bound : int
        Upper bound for timeseries
    outfile : string
        File name
    Returns
    ----------
    results : plotted timeseries
         If outfile in arguments, the plot is saved.  Otherwise, it the plot is shown.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    sns.set_style("white")
    time_data_df[(time_data_df['time'] > lower_bound) & (time_data_df['time'] < upper_bound)]['actual'].plot(ax=ax, color='k', lw=1, fontsize=21)
    time_data_df[(time_data_df['time'] > lower_bound) & (time_data_df['time'] < upper_bound)]['predicted'].plot(ax=ax, color='r', lw=1)
    ax.legend(['Actual', 'Predicted'], fontsize=21)
    ax.set_xlabel("Time", fontsize=21)
    ax.set_ylabel("Voltage (normalized)", fontsize=21)

    xvals = ax.get_xticks()
    ax.set_xticklabels([np.round(x, 4) for x in xvals])

    ax.set_xticklabels([np.round(x / 400, 4) for x in xvals])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def plot_Y_electrode(time_data, bo, lower_bound, upper_bound, electrode, outfile=None):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    sns.set_style("white")
    mask = (time_data['time'] > lower_bound) & (time_data['time'] < upper_bound)
    Y = stats.zscore(bo.get_data())
    added = mat.repmat(1 + np.arange(Y.shape[1])*2, Y.shape[0], 1)
    Y = pd.DataFrame(Y+ added)
    ax = Y[Y.columns][mask].plot(legend=False, title=None, fontsize=21, color='k', lw=.6)
    Y[Y.columns[int(electrode)]][mask].plot(color='b', lw=.8)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    yvals = ax.get_yticks()
    ax.set_yticklabels([int(np.round(x / 2, 1)) for x in yvals])
    xvals = ax.get_xticks()
    ax.set_xticklabels([np.round(x / bo.sample_rate[0], 4) for x in xvals])
    ax.set_frame_on(False)
    ax.set_xlabel("Time", fontsize=21)
    ax.set_ylabel("Electrodes", fontsize=21)
    ax.set_ylim([-4, len(Y.columns) * 2 + 4])
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()

def plot_density(df, outfile = None):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    sns.set_style("white")
    mybins =np.linspace(0, 1, 200)
    g = sns.JointGrid('Density', 'Correlation', df, xlim=[-.2,.8],ylim=[-1,1])
    g.ax_marg_x.hist(df['Density'], bins=mybins, color = 'k', alpha = .3)
    g.ax_marg_y.hist(df['Correlation'], bins=np.arange(-1, 1, .01), orientation='horizontal', color = 'k', alpha = .3)
    #g.ax_marg_x.set_xscale('log')
    g.ax_marg_x.set_xscale('linear')
    g.ax_marg_y.set_yscale('linear')
    g.plot_joint(plt.scatter, color='black', edgecolor='black', alpha = .6)
    ax = g.ax_joint
    left, width = .05, .5
    bottom, height = .1, .5
    rstat = stats.pearsonr(df['Density'], r2z(df['Correlation']))
    ax.text(left, bottom, 'r = ' + str(np.round(rstat[0],2)) + '   p < '+ str(10**-10),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=18)
    ax.set_xlabel("Density", fontsize=21)
    ax.set_ylabel("Correlation", fontsize=21)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    plt.tight_layout()

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()

def r2z(r):
    return 0.5 * (np.log(1 + r) - np.log(1 - r))

def z2r(z):
    r = np.divide((np.exp(2*z) - 1), (np.exp(2*z) + 1))
    r[np.isnan(r)] = 0
    r[np.isinf(r)] = np.sign(r)[np.isinf(r)]
    return r

def compile_df_locs(df_column):

    R_locs = []

    for i, e in enumerate(df_column):

        R = np.array(e[1:-1].split())
        R = np.atleast_2d(np.array(R.astype(np.float)))
        if R_locs == []:
            R_locs = R
        else:

            R_locs = np.vstack((R_locs, R))

    return R_locs

def draw_bounds(ax, model):
    bounds = np.where(np.diff(np.argmax(model.segments_[0], axis=1)))[0]
    bounds_aug = np.concatenate(([0],bounds,[model.segments_[0].shape[0]]))
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle((bounds_aug[i], bounds_aug[i]), bounds_aug[i+1]-bounds_aug[i],
                                 bounds_aug[i+1]-bounds_aug[i], linewidth=1, edgecolor='#FFF9AE',
                                 facecolor='none')
        ax.add_patch(rect)
    return ax


def _close_all():
    figs = plt.get_fignums()
    for f in figs:
        plt.close(f)


def _plot_borderless_clustered(x, factor_bounds=None, factor_colors=None, savefile=None, vmin=-1, vmax=1, width=1000, dpi=100, cmap='Spectral_r'):
    _close_all()
    width *= (1000.0 / 775.0)  # account for border
    height = (775.0 / 755.0) * float(width) * float(x.shape[0]) / float(x.shape[1])  # correct height/width distortion

    fig = plt.figure(figsize=(width / float(dpi), height / float(dpi)), dpi=dpi)

    if len(x.shape) == 2:
        plt.pcolormesh(x, vmin=float(vmin), vmax=float(vmax), cmap=cmap)
    else:
        plt.imshow(x)
    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    if factor_bounds:

        ax.hlines(factor_bounds, *ax.get_xlim(), linewidth=2, colors='w')
        ax.vlines(factor_bounds, *ax.get_ylim(), linewidth=2, colors='w')

    if factor_colors:
        for i, f in enumerate(factor_bounds[:-1]):
            rect = patches.Rectangle((f,f), factor_bounds[i+1]-f-1, factor_bounds[i+1]-f-1, linewidth=6, edgecolor=factor_colors[i],facecolor='none', zorder=2)
            ax.add_patch(rect)


    fig.set_frameon(False)

    if not savefile == None:
        fig.savefig(savefile, figsize=(width / float(dpi), height / float(dpi)), bbox_inches='tight', pad_inches=0,
                    dpi=dpi)
    return fig

def normalizeRows(M):
    row_sums = M.sum(axis=1)
    return M / row_sums[:, np.newaxis]

def interp_corr_old(locs, corrs, width=10, vox_size=10, outfile=None):
    nii = se.load('std', vox_size=vox_size)
    full_locs = nii.get_locs().values
    W = np.exp(_log_rbf(full_locs, locs, width=width))
    interp_corrs =  np.dot(corrs, W.T)
    bo_nii = se.Brain(data=interp_corrs, locs=full_locs)
    nii_bo = _brain_to_nifti(bo_nii, nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=1, vmin=0)
    #ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=1, vmin=0, display_mode='lyrz')

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()

def interp_corr(locs, corrs, width=10, vox_size=10, outfile=None):
    nii = se.load('std', vox_size=vox_size)
    full_locs = nii.get_locs().values
    W = np.exp(_log_rbf(full_locs, locs, width=width))
    interp_corrs = z2r(np.divide(np.dot(r2z(corrs), W.T), np.sum(W, axis=1)))
    bo_nii = se.Brain(data=interp_corrs, locs=full_locs)
    nii_bo = _brain_to_nifti(bo_nii, nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=1, vmin=0, display_mode='lyrz')
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()

def interp_density(locs, density, width=10, vox_size=10, outfile=None):
    nii = se.load('std', vox_size=vox_size)
    full_locs = nii.get_locs().values
    W = np.exp(_log_rbf(full_locs, locs, width=width))
    interp_corrs =  np.dot(density, W.T)
    bo_nii = se.Brain(data=interp_corrs, locs=full_locs)
    nii_bo = _brain_to_nifti(bo_nii, nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=.02, vmin=0, display_mode='lyrz')

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()



def gkern(kernlen=100, std=10):
    """Returns a 2D Gaussian kernel array for bullseye"""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def density(n_by_3_Locs, nearest_n, tau=.2):
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
    return np.exp(-tau*(distances.sum(axis=1) / (np.shape(distances)[1] - 1)))
#     nbrs = NearestNeighbors(n_neighbors=nearest_n, algorithm='ball_tree').fit(n_by_3_Locs)
#     distances, indices = nbrs.kneighbors(n_by_3_Locs)
#     return np.exp(-(distances.sum(axis=1) / (np.shape(distances)[1] - 1)))
