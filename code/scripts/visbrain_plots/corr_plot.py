
import supereeg as se
import os
from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj

cmap = "hot_r"

nii_bo_dir = '../../../data/niis'

sc = SceneObj(bgcolor='white', size=(1000, 500))


# Colorbar default arguments
CBAR_STATE = dict(cbtxtsz=20, txtsz=20., txtcolor='black', width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=2)
bo = se.load('/Users/lucyowen/repos/supereeg_paper/data/niis/corrmap/ram_raw_corrmap.bo')

template_brain = 'B3'

data1 = bo.get_data().values.ravel()
xyz1 = bo.locs.values


s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
s_obj_1.color_sources(data=data1)


s_obj_all = s_obj_1

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_all, clim=(0, 1), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)


b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_all, clim=(0, 1), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=0, col=1, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_all, clim=(0, 1), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=0, col=2, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_all, clim=(0, 1), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=0, col=3, rotate='right', use_this_cam=True)

cb_proj = ColorbarObj(b_obj_proj_right,
                      cblabel='Correlation',
                      cmap='hot_r',
                      vmin = 0,
                      vmax=1,
                      **CBAR_STATE)

sc.add_to_subplot(cb_proj, row=0, col=4, width_max=100)

sc.preview()
#sc.screenshot(os.path.join('/Users/lucyowen/Desktop/freqs_plots', 'all_r.png'), transparent=True)

#sc.record_animation(os.path.join('/Users/lucyowen/Desktop','freqs.gif'), n_pic=40)