import os
import supereeg as se
import hypertools as hyp
import pandas as pd
import numpy as np
import glob as glob
from scipy import stats, signal
import scipy.spatial as spatial
import numpy.matlib as mat
import matplotlib.patches as patches
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors
from nilearn import plotting as ni_plt
from supereeg.helpers import _log_rbf, _brain_to_nifti, _plot_borderless
from supereeg.helpers import _corr_column, get_rows, known_unknown
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
import math

############## General functions #################

def r2z(r):
    return 0.5 * (np.log(1 + r) - np.log(1 - r))

def z2r(z):
    r = np.divide((np.exp(2*z) - 1), (np.exp(2*z) + 1))
    try:
        r[np.isnan(r)] = 0
        r[np.isinf(r)] = np.sign(r)[np.isinf(r)]
    except:
        pass

    return r

def remove_electrode(subkarray, subarray, electrode):
    """
        Removes electrode from larger array

        Parameters
        ----------
        subkarray : ndarray
            Subject's electrode locations that pass the kurtosis test

        subarray : ndarray
            Subject's electrode locations (all)

        electrode : str
            Index of electrode in subarray to remove

        Returns
        ----------
        results : ndarray
            Subject's electrode locations that pass kurtosis test with electrode removed

        """
    rm_ind = get_rows(subkarray, subarray[electrode])
    other_inds = [i for i in range(np.shape(subkarray)[0]) if i != electrode]
    return np.delete(subkarray, rm_ind, 0), other_inds

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


############## Methods ##########################

def plot_hist(dataframe, X='Correlation', title=None, outfile=None):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    sns.set_style("white")
    df_corrs = pd.DataFrame()
    df_corrs[X] = dataframe[X].values
    n_count = len(df_corrs)
    bin_values = np.arange(start=-1, stop=1, step=.025)
    ax = sns.distplot(df_corrs, hist=True, kde=True,
             bins=bin_values, color = 'k',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

    vals = ax.get_yticks()
    ax.set_yticklabels([np.round(x/n_count,5) for x in vals])
    ax.set_ylabel('Proportion of electrodes', fontsize=21)
    ax.set_xlabel(X, fontsize=21)
    ax.set_xlim(-1, 1)
    ax.set_frame_on(False)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    #plt.text(2,10, 'mean = '+ str(np.round(dataframe['Correlation'].mean(),3)))
    left, width = .05, .5
    bottom, height = .75, .5
    plt.tight_layout()

    ax.text(left, bottom, 'mean = '+ str(np.round(dataframe[X].mean(),3)),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=18)

    if outfile:
        plt.savefig(outfile)
        plt.clf()


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

    data = time_data_df.loc[(time_data_df['time'] >= lower_bound) & (time_data_df['time'] < upper_bound)]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    sns.set_style("white")

    plt.plot(data['time'], data['predicted'], 'r')
    plt.plot(data['time'], data['actual'], 'b')
    plt.legend(['Reconstructed','Observed'], fontsize=21)
    plt.gca().set_xticks(np.linspace(lower_bound, upper_bound, np.round(upper_bound-lower_bound)+1))
    plt.xlabel('Time (s)', fontsize=21)
    plt.ylabel('Voltage (normalized)', fontsize=21)
    plt.gca().tick_params(axis='x', labelsize=18)
    plt.gca().tick_params(axis='y', labelsize=18)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
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


def gkern(kernlen=100, std=10):
    """Returns a 2D Gaussian kernel array for bullseye"""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

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


############## Corrmap #################

def plot_2_histograms(df, X, Y, xticks=True, legend=True, outfile=None):

    grouped_results = df.groupby('Subject')[Y, X].mean()
    one_sided_t_across = stats.ttest_1samp(grouped_results[Y],  0)
    one_sided_t_within = stats.ttest_1samp(grouped_results[X],  0)

    print('one sided t test for across: ')
    print(one_sided_t_across)


    print('one sided t test for within: ')
    print(one_sided_t_within)

    t_stat_group = stats.ttest_rel(grouped_results[Y],grouped_results[X])

    fig = plt.gcf()
    fig.set_size_inches(18.5, 8.5)
    fig.subplots_adjust(left=0.2, bottom=.2)
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

    ax.set_xlim(-1, 1)

    #ax.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    vals = ax.get_yticks()
    ax.set_yticklabels([np.round(x/n_count*10000,2) for x in vals])
    f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    fmt = mtick.FuncFormatter(g)

    t_left, t_width = 0, .5
    t_bottom, t_height = 1, .5
    ax.text(t_left, t_bottom, 'x' + str(fmt(10**-4)),
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes, fontsize=40)

    if legend:
        leg = ax.legend(['Across', 'Within'], fontsize=50, loc='upper left')
        LH = leg.legendHandles
        LH[0].set_color('k')
        LH[0].set_alpha(1)
        LH[1].set_color('lightgray')
        LH[1].set_alpha(1)

    ylim = ax.get_ylim()[1]
    m1, n1 = [z2r(df[X]).mean(), z2r(df[Y]).mean()], [ylim, ylim]
    plt.plot(m1, n1, marker = '|', mew=4, markersize=20, color='k', linewidth=4)
    a1 = (z2r(df[X]).mean() + z2r(df[Y]).mean()) /2
    #ax.plot(a1, ylim + .1, marker = '*', markersize=20, color='k')
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel('Proportion \n of electrodes', fontsize=50)
    
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=9))

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
    #plt.tight_layout()
    bbox_inches='tight'


    if outfile:
        plt.savefig(outfile)

    print('t test: ')
    print(t_stat_group)

def interp_corr(locs, corrs, width=10, vox_size=10, outfile=None, save_nii=None):
    nii = se.load('std', vox_size=vox_size)
    full_locs = nii.get_locs().values
    W = np.exp(_log_rbf(full_locs, locs, width=width))
    interp_corrs =  np.dot(corrs, W.T)
    bo_nii = se.Brain(data=interp_corrs, locs=full_locs)
    nii_bo = _brain_to_nifti(bo_nii, nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=1, vmin=0)
    #ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=1, vmin=0, display_mode='lyrz')

    if save_nii:
        nii_bo.save(save_nii)
        
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


############ Density ########################


def density_within_r(locs, r):
    point_tree = spatial.cKDTree(locs)
    density_locs = np.array([])

    for l in locs:
        density_locs = np.append(density_locs, np.divide(len(point_tree.query_ball_point(l, r)), np.shape(locs)[0]))

    return density_locs


def density_within_r_plot(locs, r, vox_size=4, outfile=None):

    nii = se.load('std', vox_size=vox_size)
    full_locs = nii.get_locs().values
    point_tree = spatial.cKDTree(locs)
    density_locs = np.array([])

    for l in locs:
        density_locs = np.append(density_locs, np.divide(len(point_tree.query_ball_point(l, r)), np.shape(locs)[0]))

    bo_nii = se.Brain(data=np.atleast_2d(density_locs), locs=locs)
    nii_bo = se.helpers._brain_to_nifti(bo_nii, nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=.1, vmin=0)

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def density_by_voxel(locs, r=20, vox_size=4):

    sub_nii = se.load('std', vox_size=4)
    sub_locs = sub_nii.get_locs().values

    point_tree = spatial.cKDTree(locs)
    density_locs = np.array([])

    for l in sub_locs:
        density_locs = np.append(density_locs, np.divide(len(point_tree.query_ball_point(l, r)), np.shape(locs)[0]))

    return density_locs

def density_by_voxel_plot(locs, r=20, vox_size=4, outfile=None, save_nii=None):

    sub_nii = se.load('std', vox_size=4)
    sub_locs = sub_nii.get_locs().values

    point_tree = spatial.cKDTree(locs)
    density_locs = np.array([])

    for l in sub_locs:
        density_locs = np.append(density_locs, np.divide(len(point_tree.query_ball_point(l, r)), np.shape(locs)[0]))

    bo_nii = se.Brain(data=np.atleast_2d(density_locs), locs=sub_locs)
    nii_bo = se.helpers._brain_to_nifti(bo_nii, sub_nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=.1, vmin=0, display_mode='lyrz')
    
    if save_nii:
        nii_bo.save(save_nii)
        
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


def plot_2d_hist(df, outfile=None):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 18.5)
    sns.set_style("white")
    g = (sns.jointplot('RAM', 'PyFR', df, xlim=[-.005,.025],ylim=[-.005,.025], kind="kde", color='k', height=8).set_axis_labels('RAM', 'PyFR', fontsize=30))
    ax = g.ax_joint

    left, width = .35, .5
    bottom, height = .1, .5
    rstat = stats.pearsonr(df['RAM'], df['PyFR'])
    ax.text(left, bottom, 'r = ' + str(np.round(rstat[0],2)) + '   p < '+ str(10**-10),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=24)
    ax.set_xlabel("Dataset 1 \n density by voxel", fontsize=30)
    ax.set_ylabel("Dataset 2 \n density by voxel", fontsize=30)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    plt.tight_layout()
    print(rstat)

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
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=.05, vmin=0, display_mode='lyrz')

    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


######### Best locs ##############
def most_informative_locs_plot(df, vox_size=5, width=10, outfile=None):

    locs = compile_df_locs(df['R'])

    sub_nii = se.load('std', vox_size=vox_size)
    sub_locs = sub_nii.get_locs().values

    point_tree = spatial.cKDTree(locs)


    most_info = np.array([])

    z_df = df.copy(deep=True)
    z_df['Correlation'] = r2z(z_df['Correlation'])
    for l in sub_locs:
        most_info = np.append(most_info, z_df['Correlation'][point_tree.query_ball_point(l, width)].mean())

    bo_nii = se.Brain(data=np.atleast_2d(z2r(most_info)), locs=sub_locs)
    nii_bo = se.helpers._brain_to_nifti(bo_nii, sub_nii)
    ni_plt.plot_glass_brain(nii_bo, colorbar=True, threshold=None, vmax=1, vmin=0, display_mode='lyrz')
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()

def most_informative_locs(df, vox_size=5, width=10):

    locs = compile_df_locs(df['R'])
    #locs = df['R']
    sub_nii = se.load('std', vox_size=vox_size)
    sub_locs = sub_nii.get_locs().values

    point_tree = spatial.cKDTree(locs)


    most_info = np.array([])

    z_df = df.copy(deep=True)
    z_df['Correlation'] = r2z(z_df['Correlation'])
    for l in sub_locs:

        most_info = np.append(most_info, z_df['Correlation'][point_tree.query_ball_point(l, width)].mean())

    return z2r(most_info)



def plot_2d_corr_hist(df, outfile=None):

    fig = plt.gcf()
    fig.set_size_inches(18.5, 18.5)
    sns.set_style("white")

    g = (sns.jointplot('RAM', 'PyFR', df, kind="kde", color='k', height=8).set_axis_labels('RAM', 'PyFR', fontsize=30))
    ax = g.ax_joint

    left, width = .35, .5
    bottom, height = .1, .5
    rstat = stats.pearsonr(df['RAM'], df['PyFR'])
    ax.text(left, bottom, 'r = ' + str(np.round(rstat[0],2)) + '   p < '+ str(10**-10),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=24)
    ax.set_xlabel("Dataset 1 \n per-voxel information scores", fontsize=30)
    ax.set_ylabel("Dataset 2 \n per-voxel information scores", fontsize=30)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))

    plt.tight_layout()
    print(rstat)
    if not outfile is None:
        plt.savefig(outfile)
    else:
        plt.show()


######### Supplemental ###########

def p_level(p):
    if .05 <= p <.1:
        return('+')
    if .01 <= p < .05:
        return('*')
    if .001 <= p < .01:
        return('**')
    if .0001 <= p < .001:
        return('***')
    if .00001 <= p < .0001:
        return('d')
    if .000001 <= p < .00001:
        return('dd')
    if p < .000001:
        return('ddd')

def long_form_df(full_df, df_col):

    long_form = pd.DataFrame()
    long_form['Correlation'] = full_df[df_col]
    long_form['Subject'] = df_col.split('_')[1]
    long_form['Experiment'] = df_col.split('_')[2]

    return long_form


def plot_split_violin(df, legend=True, yticks=True, outfile=None):
    fig = plt.gcf()
    fig.set_size_inches(12.5, 10.5)
    plt.ylim(-1.2,2.0)
    ax = sns.violinplot(x="Experiment", y="Correlation", hue="Subject", order=["w", "a", "all"], ylim=[-1,1], data=df, palette="gray", split=True)
    if legend:
        handles, labels = fig.get_axes()[0].get_legend_handles_labels()
        fig.get_axes()[0].legend(handles, ['Across', 'Within'], loc='lower right', fontsize=30)
    else:
        ax.legend().set_visible(False)
    ax.tick_params(axis='x', length=0, labelsize=18)
    ax.tick_params(axis='y', which='both', length=0, labelsize=18)
    ax.set_xticklabels(['Within', 'Across','All'])
    ax.set_xlabel('Experiment', fontsize=30)

    z_df = df.copy(deep=True)
    z_df['Correlation'] = r2z(z_df['Correlation'])
    yposlist_df = z2r(z_df.groupby(['Experiment', 'Subject'])['Correlation'].mean())
    yposlist = yposlist_df.reindex(["w", "a", "all"], level='Experiment').tolist()
    xposlist = range(3)

    c = 0
    for i in range(len(yposlist)):

        if (i%2) == 0:
            ax.text(c-.12, yposlist[i], np.round(yposlist[i],2), fontsize=14, color='white')
        else:
            ax.text(c+.02, yposlist[i], np.round(yposlist[i],2) , fontsize=14, color='white')
            c +=1

    ### first two significance lines:
    ylim = 1.2
    for l in range(3):
        plt.plot([l-.05, l+.05], [1.15, 1.15], marker = '|', mew=2, markersize=10, color='k', linewidth=2)
        ax.plot(l, 1.15 + .05, marker = '*', markersize=10, color='k')

        ylim += .15
        for i in range(2):
            ylim += i/14


            if l==2:
                m1, n1 = [l, l-l], [ylim, ylim]
                a1 = (l + (l-l)) /2
            else:
                m1, n1 = [l, l+1], [ylim, ylim]
                a1 = (l + (l+1)) /2
            if i == 0:
                plt.plot(m1, n1, marker = '|', mew=2, markersize=10, color='lightgray', linewidth=2)
                ax.plot(a1, ylim-.1 + .05, marker = '*', markersize=10, color='lightgray')
            else:
                plt.plot(m1, n1, marker = '|', mew=2, markersize=10, color='darkgray', linewidth=2)
                ax.plot(a1, ylim + .05, marker = '*', markersize=10, color='darkgray')



    if not yticks:
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax.set_ylabel('')
        ax.axes.get_yaxis().set_ticks([])
    else:
        ax.set_ylabel('Correlation', fontsize=30)
        for index, label in enumerate(ax.yaxis.get_ticklabels()):
            if index > 5:
                print(index)
                label.set_visible(False)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)


def plot_contour(dataframe, X, Y, title=None, outfile=None):

    sns.set_style("white")

    g = (sns.jointplot(x=X, y=Y,data=dataframe, kind="reg", color='k',
                       height=8).set_axis_labels(X, 'z-transformed correlation', fontsize=24))
    ax = g.ax_joint
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 1.5)
    #ax = plt.gca()
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 2 != 0:
            label.set_visible(False)
    for index, label in enumerate(ax.yaxis.get_ticklabels()):
        if index % 2 == 0:
            label.set_visible(False)
    left, width = .35, .5
    bottom, height = .05, .5
    rstat = stats.pearsonr(dataframe[X], dataframe[Y])
    ax.text(left, bottom, 'r = ' + str(np.round(rstat[0],2)) + ', p = '+ str(np.round(rstat[1],2)),
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes, fontsize=24)
    print(rstat)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)

def plot_split_violin_colors(df, X, Y, H, x_order, hue_order, left_colors, right_colors, latex_x=None, outfile=None):
    
    """
    X, Y, H :  names of variables in df
        Inputs for plotting long-form data
        
    df :  DataFrame
        Longform dataframe 
    
    hue_order : lists of strings
        Order to plot the categorical hue variable
        
    x_order : lists of strings
        Order to plot the categorical x variable
        
    left_colors, right_colors : RGB or RGBA tuple of float values 
        RGB values
    
    outfile : string
        Save file

    """
    

    fig, axes = plt.subplots(1, len(x_order), figsize=(30, 16), sharey='all')


    for e, freq in enumerate(x_order):

        #freq_df = pd.read_csv(os.path.join(freq_dir, freq + '.csv'), index_col=0)
        df_temp = df[df[X]==freq]

        plt.ylim(-1,1.2)
        sns.violinplot(x=X, y=Y, hue=H, ylim=[-1,1], data=df_temp, 
                       palette={hue_order[0]: left_colors[e], hue_order[1]: right_colors[e]}, split=True, ax=axes[e])
        axes[e].legend().set_visible(False)
        axes[e].tick_params(axis='x', labelsize=30)
        if latex_x: 
            axes[e].set_xticklabels([latex_x[e]])
        else:
            axes[e].set_xticklabels([freq])
        axes[e].set_xlabel('')
        axes[e].set_ylabel('')
        if e ==0:
            axes[e].tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
        else:
            axes[e].tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        axes[e].spines['right'].set_visible(False)
        axes[e].spines['left'].set_visible(False)
        axes[e].spines['top'].set_visible(False)
        axes[e].spines['bottom'].set_visible(False)

        z_df = df.copy(deep=True)
        z_df[Y] = r2z(z_df[Y])
        yposlist = z2r(z_df.groupby([X, H])[Y].mean())
        xposlist = range(1)

#         c = 0
#         for i in range(len(yposlist)):

#             if (i%2) == 0:
#                 axes[e].text(c-.13, yposlist[i], np.round(yposlist[i],2), fontsize=14, color='k')
#             else:
#                 axes[e].text(c+.05, yposlist[i], np.round(yposlist[i],2) , fontsize=14, color='k')
#                 c +=1
                
    axes[0].tick_params(axis='y', which='both', length=0, labelsize=18)
    axes[0].set_ylabel(Y, fontsize=30)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
        

def plot_split_barplot_colors(df, X, Y, H, x_order, hue_order, left_colors, right_colors, latex_x=None, outfile=None):
    
    """
    X, Y, H :  names of variables in df
        Inputs for plotting long-form data
        
    df :  DataFrame
        Longform dataframe 
    
    hue_order : lists of strings
        Order to plot the categorical hue variable
        
    x_order : lists of strings
        Order to plot the categorical x variable
        
    left_colors, right_colors : RGB or RGBA tuple of float values 
        RGB values
    
    outfile : string
        Save file

    """
    
    sns.set(style="white")

    fig, axes = plt.subplots(1, len(x_order), figsize=(13, 8), sharey='all')


    for e, freq in enumerate(x_order):
        
        #freq_df = pd.read_csv(os.path.join(freq_dir, freq + '.csv'), index_col=0)
        df_temp = df[df[X]==freq]
        #offset = long_form_freq['Correlation'].min()
        offset = -.2
        df_temp['Correlation'] = df_temp['Correlation'] - offset
        sns.barplot(x=X, y=Y, hue=H, data=df_temp, linewidth=6, alpha=1, facecolor=(1, 1, 1, 0), 
                    errcolor=left_colors[e], edgecolor=left_colors[e], palette={hue_order[0]: left_colors[e], hue_order[1]: right_colors[e]}, ax=axes[e])
        sns.stripplot(x=X, y=Y, hue=H, data=df_temp, jitter=.4, split=True, alpha=.15,
                    palette={hue_order[0]: left_colors[e], hue_order[1]: right_colors[e]}, ax=axes[e])

        axes[e].legend().set_visible(False)
        axes[e].tick_params(axis='x', labelsize=27)
        axes[e].tick_params(axis='y', labelsize=27)
        if latex_x: 
            axes[e].set_xticklabels([latex_x[e]])
        else:
            axes[e].set_xticklabels([freq])
        axes[e].set_xlabel('')
        axes[e].set_ylabel('')
        if e ==0:
            axes[e].tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
            
        else:
            axes[e].tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        axes[e].spines['right'].set_visible(False)
        axes[e].spines['left'].set_visible(False)
        axes[e].spines['top'].set_visible(False)
        axes[e].spines['bottom'].set_visible(False)

        z_df = df_temp.copy(deep=True)
        z_df[Y] = r2z(z_df[Y] + offset)
        yposlist = z2r(z_df.groupby([X, H])[Y].mean()) - offset
        xposlist = range(1)

        locs,labels = plt.yticks()

        plt.yticks(locs, map(lambda y: "%.1f" % y, locs + offset))

    axes[0].tick_params(axis='y', which='both', length=0, labelsize=27)
    axes[0].set_ylabel(Y, fontsize=27)
    axes[0].set(ylim=(-.05, 1.2))
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
        
        
def plot_ridges(df, X, H, hue_solid, hue_transparent, palette, outfile=None):
    
    """
    X, H :  names of variables in df
        Inputs for plotting wide-form data
        
    df :  DataFrame
        Longform dataframe 
    
    hue_solid, hue_transparent : strings
        How to plot the categorical hue variable
        
        
    left_colors, right_colors : RGB or RGBA tuple of float values 
        RGB values
    
    outfile : string
        Save file

    """
    
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


    g = sns.FacetGrid(df, row=H, hue=H, aspect=10, height=.99, palette=palette)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, hue_solid, clip_on=True, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, hue_solid, clip_on=True, lw=2, bw=.2)
    g.map(sns.kdeplot, hue_transparent, clip_on=True, shade=True, alpha=.2, lw=1.5, bw=.2)
    g.map(sns.kdeplot, hue_transparent, clip_on=True, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=True)

    first_freq = df[H].unique()[0]
    yposlist = z2r(df.groupby([H])[hue_solid].mean()[first_freq])

    g.map(plt.axvline, x=yposlist, lw=2, color="k", clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, hue_solid)
    g.set(xlim=(-.5,1.2))
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.15)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.set(xticks=[-.2, 0.0, .2, .4, .6, .8, 1.0])
    plt.xlabel(X, fontsize=16)
    g.despine(bottom=True, left=True)
    if outfile:
        plt.savefig(outfile)
        
        