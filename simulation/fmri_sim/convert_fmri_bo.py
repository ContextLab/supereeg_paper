
import supereeg as se
import os
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiMasker
import nibabel as nib
import warnings
from config import config


def nii2cmu(nifti_file, mask_file=None):
    def fullfact(dims):
        '''
        Replicates MATLAB's fullfact function (behaves the same way)
        '''
        vals = np.asmatrix(range(1, dims[0] + 1)).T
        if len(dims) == 1:
            return vals
        else:
            aftervals = np.asmatrix(fullfact(dims[1:]))
            inds = np.asmatrix(np.zeros((np.prod(dims), len(dims))))
            row = 0
            for i in range(aftervals.shape[0]):
                inds[row:(row + len(vals)), 0] = vals
                inds[row:(row + len(vals)), 1:] = np.tile(aftervals[i, :], (len(vals), 1))
                row += len(vals)
            return inds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        img = nib.load(nifti_file)
        mask = NiftiMasker(mask_strategy='background')
        if mask_file is None:
            mask.fit(nifti_file)
        else:
            mask.fit(mask_file)

    hdr = img.header
    S = img.get_sform()
    vox_size = hdr.get_zooms()
    im_size = img.shape

    if len(img.shape) > 3:
        N = img.shape[3]
    else:
        N = 1

    Y = np.float64(mask.transform(nifti_file)).copy()
    vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
    vox_coords = fullfact(img.shape[0:3])[vmask, ::-1] - 1

    R = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]

    null_inds = ~np.all(Y == Y[0, :], axis=0)


    Y = Y[:, null_inds]
    R = R[null_inds]

    return Y, R

results_dir = config['bof_datadir']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

fmri_dir = config['fmri_datadir']

nii = os.path.join(config['locs_resultsdir'], 'gray_3.nii')


for i in list(range(1, len(os.listdir(config['fmri_datadir']))+1)):


    bo_file = os.path.join(results_dir, 'sub-%d' % i + '.bo')

    if not os.path.exists(bo_file):

        try:
            ## need to do this for intact1 and intact 2!

            data, locs = nii2cmu(os.path.join(fmri_dir, 'sherlock_movie_s%d' % i + '.nii'), mask_file=nii)
            bo = se.Brain(data=data, locs=locs, sample_rate=1)
            bo.save(bo_file)
            print(bo.get_locs().shape)
        except:
            print(bo_file + '_issue')



print('done converting brain objects')