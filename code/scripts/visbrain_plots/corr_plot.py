
import supereeg as se
import os
import glob as glob
from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj

cmap = "hot_r"

nii_bo_dir = '../../../data/niis/corrmap'
fig_dir = '../../../paper/figs/source/corrmap'

r = 10

CBAR_STATE = dict(cbtxtsz=20, txtsz=20., txtcolor='black', width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=2)

template_brain = 'B3'

bo_list = glob.glob(os.path.join(nii_bo_dir, '*_raw_*.bo'))

for b in bo_list:

    bo = se.load(b)

    fname = os.path.splitext(os.path.basename(b))[0]

    sc = SceneObj(bgcolor='white', size=(1000, 1000))

    data1 = bo.get_data().values.ravel()
    xyz1 = bo.locs.values


    s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
    s_obj_1.color_sources(data=data1)


    s_obj_all = s_obj_1

    b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_left.project_sources(s_obj_all, clim=(0, 1), cmap=cmap, radius=r)
    sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)


    b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_left.project_sources(s_obj_all, clim=(0, 1), cmap=cmap, radius=r)
    sc.add_to_subplot(b_obj_proj_left, row=1, col=0, rotate='right', use_this_cam=True)

    b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_right.project_sources(s_obj_all, clim=(0, 1), cmap=cmap, radius=r)
    sc.add_to_subplot(b_obj_proj_right, row=1, col=1, rotate='left', use_this_cam=True)

    b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_right.project_sources(s_obj_all, clim=(0, 1), cmap=cmap, radius=r)
    sc.add_to_subplot(b_obj_proj_right, row=0, col=1, rotate='right', use_this_cam=True)

    cb_proj = ColorbarObj(b_obj_proj_right,
                          cblabel='Correlation',
                          cmap='hot_r',
                          vmin = 0,
                          vmax=1,
                          **CBAR_STATE)

    #sc.add_to_subplot(cb_proj, row=0, col=2, width_max=100)

    sc.screenshot(os.path.join(fig_dir, fname + '.pdf'), transparent=True)
