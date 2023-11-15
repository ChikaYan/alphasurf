import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import json
import imageio
from pathlib import Path
from PIL import Image
from matplotlib import rc
import string

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
rc('text', usetex=True)

class ImageSet:
    def __init__(
        self, imgs_path, img_ext='png', 
        verbose=True, white_bg=True, 
        img_name_pattern=None, 
        zoom=None, img_size=None,
        patch=None, # draw a red box around the img. ((left, top), (wdith, height))
        patch_zoom=1.5, # zoom patch and place it somewhere
        patch_loc = 'bl', # bottom left, top left, bottom right top right
        take_patch=None, # take specified patch and zoom to img_size. Given in same format of patch
    ):
        self.imgs_path = Path(imgs_path)
        self.verbose = verbose
        self.img_ext = img_ext
        self.white_bg = white_bg
        self.img_name_pattern = img_name_pattern
        self.zoom = zoom
        self.img_size = img_size
        self.patch = patch
        self.patch_zoom = patch_zoom
        self.patch_loc = patch_loc
        self.take_patch = take_patch
        
    def draw_box(self, img, left, right, top, bottom, line_width=10, line_color=(1,0,0)):
        patch = np.copy(img[top+line_width:bottom-line_width, left+line_width: right-line_width,:])
        img[top:bottom, left: right,:] = np.array(line_color)
        img[top+line_width:bottom-line_width, left+line_width: right-line_width,:] = patch
        return img
    
    
    def zoom_img(self, img, zoom=None, target_wh=None):
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        original_size = img.size
        if target_wh is not None:
            new_size = target_wh
        else:
            new_size = (int(original_size[0] * zoom), int(original_size[1] * zoom))
        img = img.resize(new_size, Image.LANCZOS)
        return np.array(img) / 255.
        
        
    def __len__(self):
        return len(list(self.imgs_path.glob(f'*.{self.img_ext}')))
        
    def __getitem__(self, key):
        if self.img_name_pattern is None:
            im_path = list(self.imgs_path.glob(f'*.{self.img_ext}'))
            im_path = sorted(im_path)

            if key >= len(im_path):
                raise Exception(f'Index {key} exceeds imgs in path {str(self.imgs_path)}')
            im_path = im_path[key]
        else:
            im_path = self.imgs_path / self.img_name_pattern.format(key)
        if self.verbose:
            print(f'Read: {str(im_path)}')

        im = imageio.imread(str(im_path)).astype('float') / 255

        if im.shape[-1] > 3:
            if self.white_bg:
                im = im[..., :3] * im[..., 3:] + (1.-im[..., 3:])
            else:
                im = im[..., :3] * im[..., 3:]


        if self.take_patch is not None:
            # prioritize take patch
            ((left, top), (width, height)) = self.take_patch
            bottom = top + height
            right = left + width
                        
            img_patch = np.copy(im[top:bottom, left:right,:])
            # zoom patch
            im = self.zoom_img(img_patch, target_wh=self.img_size)


        
        if self.zoom is not None and self.zoom != 1.:
            im = self.zoom_img(im, self.zoom)

    
        
        if self.img_size is not None:
            height, width = im.shape[:2]
            new_width, new_height = self.img_size
            left = (width - new_width)//2
            top = (height - new_height)//2
            right = (width + new_width)//2
            bottom = (height + new_height)//2
            im = im[top:bottom, left:right, :]
    
        if self.patch is not None:
            # draw and zoom a patch
            ((left, top), (width, height)) = self.patch
            bottom = top + height
            right = left + width
                        
            img_patch = np.copy(im[top:bottom, left:right,:])
            # zoom patch
            img_patch = self.zoom_img(img_patch, self.patch_zoom)
            
            # draw box
            im = self.draw_box(im, left, right, top, bottom)
            ph, pw, _ = img_patch.shape
            img_patch = self.draw_box(img_patch, 0, pw, 0, ph)
            
            h, w, _ = im.shape
            # place patch
            if self.patch_loc == 'bl': 
                im[h-ph:h, 0:pw, :] = img_patch
            elif self.patch_loc == 'br': 
                im[h-ph:h, w-pw:w, :] = img_patch
            elif self.patch_loc == 'ur': 
                im[0:ph, w-pw:w, :] = img_patch
            else:
                pass

        return im
    
    def get_path(self, key):
        im_path = sorted(list(self.imgs_path.glob(f'*.{self.img_ext}')))[key]
        return str(im_path)
    
    

def make_plot(scenes, methods, transpose=False,
              special_group=[], special_group_replace={}, special_index={}, scene_name_map={},
              subplot_size=(5, 5), fontsize=16, spacing=0.1, img_size=(800,800), verbose=False):
    
    nrow = len(scenes)
    ncols = len(methods)
    subplot_scale = (img_size[0] / 800., img_size[1] / 800.)
    if transpose:
        nrow, ncols = ncols, nrow
#         subplot_scale = (subplot_scale[1], subplot_scale[0])
        subplot_size = (subplot_size[1], subplot_size[0])

    fig,axes = plt.subplots(
        nrows=nrow, ncols=ncols,
#         figsize=(int(ncols * subplot_size * subplot_scale[0]),int(nrow * subplot_size*subplot_scale[1])),
        figsize=(int(ncols * subplot_size[1]),int(nrow * subplot_size[0])),
        gridspec_kw={'wspace':spacing,'hspace':spacing}
        )

    for i, scene in enumerate(scenes):
        for j, method in enumerate(methods):

            imgs_path = methods[method]

            if scene in special_group and method in special_group_replace:
                imgs_path = special_group_replace[method]

            # kwargs = {}
            kwargs = {'zoom': 1.25}
            if scene == 'scale':
                kwargs['zoom'] = 1.75
                kwargs['patch'] = ((150, 220), (80, 80))
                kwargs['patch_zoom'] = 4
                kwargs['patch_loc'] = 'br'
            elif scene == 'chest':
                kwargs['zoom'] = 2.5
            elif scene == 'bottle_2':
                kwargs['zoom'] = 1.75
            elif scene == 'ship_bottle':
                kwargs['zoom'] = 2

            image_set = ImageSet(
                imgs_path.format(scene), 
                verbose=verbose, 
                img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                img_size=img_size, 
                **kwargs)

            img_idx = scenes[scene]
            if method in special_index and scene in special_index[method]:
                img_idx = special_index[method][scene]

            if transpose:
                ax = axes[j, i]
            else:
                ax = axes[i, j]

            try:
                im = image_set[img_idx]
                if scene == 'lyre':
                    im = np.flip(im, axis=0)
                ax.imshow(im)
            except Exception as e:
                im = imageio.imread('/rds/project/rds-qxpdOeYWi78/plenoxels/data/empty.png') / 255.
                ax.imshow(im)
                # ax.imshow(np.ones([*img_size, 3]))
                print(e)
    #         ax.axis('off')

            # make spines (the box) invisible
            plt.setp(ax.spines.values(), visible=False)
            # remove ticks and labels for the left axis
            ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

            if not transpose:
                if j == 0:
                    scene_name = scene_name_map[scene] if scene in scene_name_map else scene
                    # scene_name.replace(scene_name[0], scene_name[0].upper(), 1)
                    ax.set_ylabel(scene_name, fontsize=fontsize)
                    
                if i == len(scenes) - 1:
                    ax.set_xlabel(method, fontsize=fontsize)
            else:
                if j == 0:
                    scene_name = scene_name_map[scene] if scene in scene_name_map else scene
                    # scene_name.replace(scene_name[0], scene_name[0].upper(), 1)
                    ax.set_title(scene_name, fontsize=fontsize)
                    
                if i == 0:
                    ax.set_ylabel(method, fontsize=fontsize)

    return fig, axes    


methods = {
    'RGB': '/rds/project/rds-qxpdOeYWi78/plenoxels/data/nerf_synthetic/{}/test',
    'GT': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/thresh_10/ckpt/imgs_gt_test',
    'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask_white/fine_mesh/imgs_pt_test',
    'HFS': '/rds/project/rds-qxpdOeYWi78/HFS/exps_1.5/{}/womask_white_no_bg/fine_mesh/imgs_pt_test',
    # r'MipNeRF360 ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/multinerf/results/blender/{}/pts/lv_10_pts/imgs_pt_test',
    r'MipNeRF360': '/rds/project/rds-qxpdOeYWi78/multinerf/results/blender/{}/pts/lv_50_pts/imgs_pt_test',
    # r'Plenoxels ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/thresh_10/ckpt/imgs_pt_test',
    r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/thresh_30/ckpt/imgs_pt_test',
    # r'Plenoxels ($\sigma=50$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/thresh_50/ckpt/imgs_pt_test',
    # r'Plenoxels ($\sigma=100$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/thresh_100/ckpt/imgs_pt_test',
    'Ours (Conv Lv)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_converge_lv_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9/ckpt_eval_surf_masked/ckpt/imgs_pt',
    'norm_l12_decay_9_low_tv_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_low_tv_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    
    # 'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc/ckpt_eval_surf_single/ckpt/imgs_pt_test',
    # 'trans': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/re_submit/trans/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'trans_high_norm': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/re_submit/trans_high_norm/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'trans_high_norm_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/re_submit/trans_high_norm_2/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'norm_l12_decay_7': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_7/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'norm_l12_decay_9': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9 (single mask)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9/ckpt_eval_surf_single_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_10': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_10/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_6': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_6/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_7': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_7/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_8/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_high_tv_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_high_tv_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_11': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_11/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_converge_lv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_converge_lv/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_converge_lv_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_converge_lv_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_converge_lv_2_no_mask': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_converge_lv_2/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_less_sparse': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_less_sparse/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_less_sparse_no_mask': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_less_sparse/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_less_sparse_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_less_sparse_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_less_sparse_2_no_mask': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_less_sparse_2/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_rescale': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_rescale/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_rescale_no_mask': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_rescale/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_rescale_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_rescale_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'norm_l12_decay_9_lv_rescale_2_no_mask': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/norm_l12/norm_l12_decay_9_lv_rescale_2/ckpt_eval_surf_single/ckpt/imgs_pt',
    
    
    
    # 'trans_high_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/re_submit/trans_high_tv/ckpt_eval_surf_single/ckpt/imgs_pt',
    # 'trans_high_tv_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/re_submit/trans_high_tv_2/ckpt_eval_surf_single/ckpt/imgs_pt',

    
    # 'MipNeRF360 (depth 0.1)': '/rds/project/rds-qxpdOeYWi78/multinerf/results/blender/{}/pts/distance_mode_01_pts/imgs_pt_test',
    # 'MipNeRF360 (depth 0.9)': '/rds/project/rds-qxpdOeYWi78/multinerf/results/blender/{}/pts/distance_mode_09_pts/imgs_pt_test',
    # 'Plenoxels (depth 0.1)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/imgs_pt_test',
    # 'Ours (trimmed)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc/ckpt_eval_surf_masked/ckpt/imgs_pt_test',
}

scenes = {
    "case": 30, #186, #30,
    "dinning_table": 168,
    # "glass_table": 54,
    "coffe_table": 30,
    "kitchen_table": 30,
    "bottle_2": 174, # 90, #66,
    # "hourglass": 30,
    # "ship_bottle": 126,
    "monkey": 0,
    "double_table": 84,
    "leaf_vase": 48,
    # "toothbrush": 114,
    # "bottle_only": 90,
    # "door": 66,
}

special_index = {
}

scene_name_map = {
    'glass_table': 'table',
    'glasses2': 'glasses',
    'lego_transparent': 'lego t',
    'bottle_2': 'bottle',
    'dinning_table':'table'
}

fig,axes = make_plot(scenes, methods, transpose=True,
                     special_index=special_index, scene_name_map=scene_name_map,
                     fontsize=25, spacing=0.02, verbose=False)

out_path = 'paper/semi_trans_new.png'
fig.savefig(out_path, facecolor='white', bbox_inches='tight', dpi=60)

print(f"Saved to {out_path}")
