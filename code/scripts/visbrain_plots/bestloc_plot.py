"""
Display a volumetric fmri-stats map across the brain surface
=============================================================
This example script does the following:
  1) Download a nifti file from nilearn data set
  2) Resample it on visbrain BrainObj
  3) Plot it in a thresholded fashion
Inspired by nilearn.plotting.plot_surf_stat_map
"""

from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj
from nilearn import datasets, surface
import nibabel as nib
import os
import glob as glob
import supereeg as se

cmap = "hot_r"

nii_bo_dir = '../../../data/niis/best_locs'
fig_dir = '../../../paper/figs/source/best_locs'

# Colorbar default arguments
CBAR_STATE = dict(cbtxtsz=20, txtsz=20., txtcolor='white', width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=2)

template_brain = 'B3'

bo = se.load(os.path.join(nii_bo_dir, 'best_90th.bo'))

data1 = bo.get_data().values.ravel()
xyz1 = bo.locs.values

s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
s_obj_1.color_sources(data=data1)

sc = SceneObj(bgcolor='white', size=(1000, 500))

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(0, 2), cmap='binary')
sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)


b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(0, 2), cmap='binary')
sc.add_to_subplot(b_obj_proj_left, row=0, col=1, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(0, 2), cmap='binary')
sc.add_to_subplot(b_obj_proj_right, row=0, col=2, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(0, 2), cmap='binary')
sc.add_to_subplot(b_obj_proj_right, row=0, col=3, rotate='right', use_this_cam=True)

sc.screenshot(os.path.join(fig_dir, 'intersection.png'), transparent=True)

nii_list = glob.glob(os.path.join(nii_bo_dir, '*_bestloc.nii'))

for n in nii_list:

    sc = SceneObj(bgcolor='white', size=(1000, 500))

    nii = nib.load(n)

    fname = os.path.splitext(os.path.basename(n))[0]

    b_obj_proj_ll = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_lr = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_rr = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_rl = BrainObj(template_brain, hemisphere='right', translucent=False)


    meshll  = [b_obj_proj_ll.vertices, b_obj_proj_ll.faces]
    meshlr  = [b_obj_proj_lr.vertices, b_obj_proj_lr.faces]

    meshrr  = [b_obj_proj_rr.vertices, b_obj_proj_rr.faces]
    meshrl  = [b_obj_proj_rl.vertices, b_obj_proj_rl.faces]

    texturell = surface.vol_to_surf(nii, meshll)
    texturelr = surface.vol_to_surf(nii, meshlr)
    texturerr = surface.vol_to_surf(nii, meshrr)
    texturerl = surface.vol_to_surf(nii, meshrl)

    texturell = texturell.ravel()
    texturelr = texturelr.ravel()
    texturerr = texturerr.ravel()
    texturerl = texturerl.ravel()


    b_obj_proj_ll.add_activation(texturell, hemisphere='left',
                         cmap='hot_r',
                         vmin=0,
                         vmax=1,
                         clim=(0, 1))


    b_obj_proj_lr.add_activation(texturelr, hemisphere='left',
                         cmap='hot_r',
                         vmin=0,
                         vmax=1,
                         clim=(0, 1))

    b_obj_proj_rr.add_activation(texturerr, hemisphere='right',
                         cmap='hot_r',
                         vmin=0,
                         vmax=1,
                         clim=(0, 1))
    b_obj_proj_rl.add_activation(texturerl, hemisphere='right',
                         cmap='hot_r',
                         vmin=0,
                         vmax=1,
                         clim=(0, 1),
                         hide_under=0)

    cb_proj = ColorbarObj(b_obj_proj_rl,
                          cblabel='Correlation',
                          cmap='hot_r',
                          vmin = 0,
                          vmax=1,
                          **CBAR_STATE)


    sc.add_to_subplot(b_obj_proj_ll, row=0, col=0, rotate='left')

    sc.add_to_subplot(b_obj_proj_lr, row=0, col=1, rotate='right')

    sc.add_to_subplot(b_obj_proj_rl, row=0, col=2, rotate='left')

    sc.add_to_subplot(b_obj_proj_rr, row=0, col=3, rotate='right')

    #sc.add_to_subplot(cb_proj, row=0, col=4, width_max=100)

    #sc.preview()

    sc.screenshot(os.path.join(fig_dir, fname + '.png'), transparent=True)