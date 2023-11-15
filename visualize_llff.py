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

        if self.take_patch is not None:
            # prioritize take patch
            ((left, top), (width, height)) = self.take_patch
            bottom = top + height
            right = left + width
                        
            img_patch = np.copy(im[top:bottom, left:right,:])
            # zoom patch
            im = self.zoom_img(img_patch, target_wh=self.img_size)
    
        
    
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
              subplot_size=(5, 3.2), fontsize=16, spacing=0.1, img_size=(500,500), verbose=False):
    
    nrow = len(scenes)
    ncols = len(methods)
    if transpose:
        nrow, ncols = ncols, nrow
#         subplot_scale = (subplot_scale[1], subplot_scale[0])
        subplot_size = (subplot_size[1], subplot_size[0])

    fig,axes = plt.subplots(
        nrows=nrow, ncols=ncols,
#         figsize=(int(ncols * subplot_size * subplot_scale[0]),int(nrow * subplot_size*subplot_scale[1])),
        figsize=(ncols * subplot_size[1],nrow * subplot_size[0] * 0.65),
        gridspec_kw={'wspace':spacing,'hspace':spacing}
        )

    for i, scene in enumerate(scenes):
        for j, method in enumerate(methods):

            imgs_path = methods[method]

            if scene in special_group and method in special_group_replace:
                imgs_path = special_group_replace[method]

            kwargs = {}
#             kwargs = {'zoom': 1.5}

            # if scene in ['trex', 'fern', 'orchids']:
            #     kwargs['zoom'] = 1.5
            #     if method == 'RGB':
            #         kwargs['zoom'] = 0.175
            
            # if scene in ['orchids']:
            #     kwargs['zoom'] = 0.8
            #     if method == 'RGB':
            #         kwargs['zoom'] = 0.175
            

            
            # if scene in ['glass_cup', 'rough', 'flower_thin', 'plastic']:
            kwargs['zoom'] = 1.2
            if method == 'RGB':
                kwargs['zoom'] = 0.17

            if scene == 'glass_cup':
                kwargs['take_patch'] = ((140, 140), (300, 300))
            elif scene == 'rough':
                kwargs['take_patch'] = ((140, 140), (250, 250))

                
            
            image_set = ImageSet(
                imgs_path.format(scene), 
                verbose=verbose, 
                # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                img_size=img_size, 
                img_ext='*',
                **kwargs)

            img_idx = scenes[scene]
            if method in special_index and scene in special_index[method]:
                img_idx = special_index[method][scene]

            if transpose:
                ax = axes[j, i]
            else:
                ax = axes[i, j]
                
#             im = image_set[img_idx]
#             ax.imshow(im)
            try:
                im = image_set[img_idx]
                ax.imshow(im)
            except Exception as e:
                ax.imshow(np.ones([*img_size, 3]))
                print(e)

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
    'RGB': '/rds/project/rds-qxpdOeYWi78/plenoxels/data/real/{}/images',
    'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask/fine_mesh/imgs_pt_train',
    'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    # 'Plenoxels (low tv)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_low_tv_2/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    # 'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # r'Plenoxels ($\sigma=50$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff/ckpt_eval_cuvol/thresh_50/imgs_pt_train',
    # r'Plenoxels ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    # r'Plenoxels No Bg ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_no_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    


    # 'l12_norm_dilate_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l12_norm_dilate_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    # 'fast_l12_norm_dilate_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_l12_norm_dilate_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',

    # 'Plenoxels bg': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    # 'fast_l2_norm_no_tv_no_sparse_d16': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/fast_l2_norm_no_tv_no_sparse_d16/ckpt_eval_surf_masked/imgs_pt_train',
    
    
    # 'synllff_low_tv_2/fast_l12_norm_dilate_4_clip': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_4_clip/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_8_high_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_8_high_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_8_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_8_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_8_high_l2_norm': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_8_high_l2_norm/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_4_no_sparse': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_4_no_sparse/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_no_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_no_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_tv_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_tv_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_tv_3': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/l12_norm_dilate_4_no_sparse_norm_tv_3/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_4_no_sparse_norm_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_4_no_sparse_norm_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_4_no_sparse_norm_tv_clip': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_4_no_sparse_norm_tv_clip/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_4_no_sparse_norm_tv_clip_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_4_no_sparse_norm_tv_clip_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_8_no_sparse_norm_tv_clip': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_8_no_sparse_norm_tv_clip/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_2_l12_norm_dilate_16_no_sparse_norm_tv_clip': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_2_l12_norm_dilate_16_no_sparse_norm_tv_clip/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_3_l12_norm_dilate_16_no_sparse_norm_tv_clip': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_3_l12_norm_dilate_16_no_sparse_norm_tv_clip/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_16_no_sparse_norm_tv_clip': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_16_no_sparse_norm_tv_clip/ckpt_eval_surf_masked/imgs_pt_train',


    # 'mid_l12_norm_dilate_4_no_bg': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_no_bg/mid_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4_no_bg': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_no_bg/mid_2_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4_high_tv_edge': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4_high_tv_edge/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4_high_tv_edge_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4_high_tv_edge_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4_high_tv_edge_3': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4_high_tv_edge_3/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4_high_tv_edge_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4_high_tv_edge_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_2_l12_norm_dilate_4_high_tv_edge_5': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4_high_tv_edge_5/ckpt_eval_surf_masked/imgs_pt_train',

    # 'l12_norm_dilate_4_no_bg': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_no_bg/l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # # 'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff/single_lv_less_trunc/ckpt_eval_surf_single/imgs_pt_train',
    # 'slv_fast_3': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff/slv_fast_3/ckpt_eval_surf_single/imgs_pt_train',
    # # 'mlv_fast': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mlv_fast/ckpt_eval_surf_masked/imgs_pt_train',
    # # 'mlv_fast_d8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mlv_fast_d8/ckpt_eval_surf_masked/imgs_pt_train',
    # # 'mlv_fast_high_norm': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_high_norm/ckpt_eval_surf_masked/imgs_pt_train',
    # # 'mlv_fast_high_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_high_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # # 'fast_l12_norm': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm/ckpt_eval_surf_masked/imgs_pt_train',
    # # 'fast_l12_norm_dilate_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    # 'fast_high_l12_norm_dilate_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_high_l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    # 'fast_l12_norm_dilate_8_high_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    # 'fast_l12_norm_dilate_8_smooth': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_8_smooth/ckpt_eval_surf_masked/imgs_pt_train',
    # 'fast_l12_norm_dilate_8_tv_edge': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_8_tv_edge/ckpt_eval_surf_masked/imgs_pt_train',
    # 'synllff_low_tv_2/fast_l12_norm_dilate_8': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/synllff_low_tv_2/fast_l12_norm_dilate_8/ckpt_eval_surf_masked/imgs_pt_train',
    
}

scenes = {
#     "trex": 0,
    # "flower_thin": 0,
    "glass_cup": 5,
    "rough": 6,
    # "half_cup" : 13,
    # "half_cup_2" : 12,
    # "bag": 10,
    # "gauze": 24,
    # "fern": 0,
    # "orchids": 0,
    # "plastic": 19,
}


special_index = {}

special_group = [
    # 'glass_cup', 
    'rough',
    # 'plastic',
#     'fern'
#     'flower_thin'
]

special_group_replace = {
#     r'Plenoxels ($\sigma=50$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_low_tv/ckpt_eval_cuvol/thresh_50/imgs_pt_train',
#     r'Plenoxels ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_low_tv/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
#     'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_low_tv/multi_lv_large_tv_fast_surf_2/ckpt_eval_surf_single/imgs_pt_train',
#     'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_low_tv/multi_lv_large_tv_fast_surf_4/ckpt_eval_surf_masked/imgs_pt_train',
    r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/fast_l2_norm_surftv_no_sparse_d16_small_tv_3/ckpt_eval_surf_masked/imgs_pt_train',
    
}

scene_name_map = {
    'flower_thin': 'gypso',
    'glass_cup': 'mug',
    # 'chair_re': 'chair',
    # 'hotdog_re': 'hotdog',
    'rough': 'cup',
}

fig,axes = make_plot(scenes, methods, transpose=False,
                     special_index=special_index, scene_name_map=scene_name_map,
                     special_group=special_group, special_group_replace=special_group_replace,
                     fontsize=20, spacing=0.02, verbose=False)


out_path = 'paper/llff.png'
out_path = 'paper/llff.pdf'
fig.savefig(out_path, facecolor='white', bbox_inches='tight', dpi=100)

print(f"Saved to {out_path}")
