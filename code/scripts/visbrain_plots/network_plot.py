
import supereeg as se
import os
from visbrain.objects import BrainObj, SceneObj, SourceObj

cmap = "yeo_colors_7_l"

template_brain = 'B3'

CBAR_STATE = dict(cbtxtsz=12, clim=[0, 7], txtsz=10., width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=1)


nii_bo_dir = '../../../data/niis/networks'

fig_dir = '../../../paper/figs/source/networks'

freqs = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma', 'broadband']

sc = SceneObj(bgcolor='white', size=(1000, 1000))

b_yeo = se.load(os.path.join(nii_bo_dir, 'yeo_bo_6mm.bo'))

data1 = b_yeo.get_data().values.ravel()
xyz1 = b_yeo.locs.values

s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
s_obj_1.color_sources(data=data1)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=1, col=0, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=1, col=1, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=0, col=1, rotate='right', use_this_cam=True)

sc.screenshot(os.path.join(fig_dir, '7_networks.png'), transparent=True)


sc = SceneObj(bgcolor='white', size=(1000, 1000))

b_yeo = se.load(os.path.join(nii_bo_dir, 'raw_network.bo'))

data1 = b_yeo.get_data().values.ravel()
xyz1 = b_yeo.locs.values

s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
s_obj_1.color_sources(data=data1)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=0, col=1, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=0, col=2, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=0, col=3, rotate='right', use_this_cam=True)

sc.screenshot(os.path.join(fig_dir,  'raw_networks.png'), transparent=True)



for f in freqs:

    sc = SceneObj(bgcolor='white', size=(1000, 1000))

    b_yeo = se.load(os.path.join(nii_bo_dir, f + '_network.bo'))

    data1 = b_yeo.get_data().values.ravel()
    xyz1 = b_yeo.locs.values

    s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
    s_obj_1.color_sources(data=data1)

    b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)

    b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_left, row=0, col=1, rotate='right', use_this_cam=True)

    b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_right, row=0, col=2, rotate='left', use_this_cam=True)

    b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_right, row=0, col=3, rotate='right', use_this_cam=True)

    sc.screenshot(os.path.join(fig_dir, f + '_networks.png'), transparent=True)
