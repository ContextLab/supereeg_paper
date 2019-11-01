"""
Display a volumetric fmri-stats map across the brain surface
=============================================================
This example script does the following:
  1) Download a nifti file from nilearn data set
  2) Resample it on visbrain BrainObj
  3) Plot it in a thresholded fashion
Inspired by nilearn.plotting.plot_surf_stat_map
"""

from visbrain.objects import BrainObj, ColorbarObj, SceneObj
from nilearn import datasets, surface
import nibabel as nib

sc2 = SceneObj(bgcolor='white', size=(1000, 500))


# Colorbar default arguments
CBAR_STATE = dict(cbtxtsz=20, txtsz=20., txtcolor='black', width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=2)

my_images = nib.load('/Users/lucyowen/repos/supereeg_paper/data/niis/density/ram_density.nii')

stat_img2 = my_images


template_brain = 'B3'


b_obj_proj_ll = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_lr = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_rr = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_rl = BrainObj(template_brain, hemisphere='right', translucent=False)


meshll  = [b_obj_proj_ll.vertices, b_obj_proj_ll.faces]
meshlr  = [b_obj_proj_lr.vertices, b_obj_proj_lr.faces]

meshrr  = [b_obj_proj_rr.vertices, b_obj_proj_rr.faces]
meshrl  = [b_obj_proj_rl.vertices, b_obj_proj_rl.faces]

texturell = surface.vol_to_surf(stat_img2, meshll)
texturelr = surface.vol_to_surf(stat_img2, meshlr)
texturerr = surface.vol_to_surf(stat_img2, meshrr)
texturerl = surface.vol_to_surf(stat_img2, meshrl)

texturell = texturell.ravel()
texturelr = texturelr.ravel()
texturerr = texturerr.ravel()
texturerl = texturerl.ravel()

b_obj_proj_ll.add_activation(texturell, hemisphere='left',
                     cmap='hot_r',
                     vmin=0,
                     vmax=.1,
                     clim=(0, .1))

b_obj_proj_lr.add_activation(texturelr, hemisphere='left',
                     cmap='hot_r',
                     vmin=0,
                     vmax=.1,
                     clim=(0, .1))

b_obj_proj_rr.add_activation(texturerr, hemisphere='right',
                     cmap='hot_r',
                     vmin=0,
                     vmax=.1,
                     clim=(0, .1))
b_obj_proj_rl.add_activation(texturerl, hemisphere='right',
                     cmap='hot_r',
                     vmin=0,
                     vmax=.1,
                     clim=(0, .1),
                     hide_under=0)

cb_proj = ColorbarObj(b_obj_proj_rl,
                      cblabel='Correlation',
                      cmap='hot_r',
                      vmin = 0,
                      vmax=.1,
                      **CBAR_STATE)


sc2.add_to_subplot(b_obj_proj_ll, row=0, col=0, rotate='left')

sc2.add_to_subplot(b_obj_proj_lr, row=0, col=1, rotate='right')

sc2.add_to_subplot(b_obj_proj_rl, row=0, col=2, rotate='left')

sc2.add_to_subplot(b_obj_proj_rr, row=0, col=3, rotate='right')

sc2.add_to_subplot(cb_proj, row=0, col=4, width_max=100)

sc2.preview()
