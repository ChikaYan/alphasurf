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
        interpolation_type=Image.LANCZOS,
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
        self.interpolation_type = interpolation_type
        
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
        # img = img.resize(new_size, Image.LANCZOS)
        # img = img.resize(new_size, Image.NEAREST)
        img = img.resize(new_size, self.interpolation_type)
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

        if self.take_patch is not None:
            # prioritize take patch
            ((left, top), (width, height)) = self.take_patch
            bottom = top + height
            right = left + width
                        
            img_patch = np.copy(im[top:bottom, left:right,:])
            # zoom patch
            im = self.zoom_img(img_patch, target_wh=self.img_size)
    
        if self.img_size is not None:
            # im = self.zoom_img(im, target_wh=self.img_size)
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
            kwargs = {'zoom': 1.4}
            kwargs['take_patch'] = (np.array([[150,200],img_size]) * kwargs['zoom']).astype(int)

            if method == 'RGB':
                kwargs['interpolation_type'] = Image.NEAREST

                if scene.endswith('_400'):
                    kwargs['zoom'] *= 2
                if scene.endswith('_200'):
                    kwargs['zoom'] *= 4

            # kwargs['take_patch'] = ((0,0),(int(800 * 1.4),int(500 * 1.4)))

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
                ax.imshow(im)
            except Exception as e:
                ax.imshow(np.ones([*img_size, 3]))
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
    # 'GT': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/ficus/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/imgs_gt_test',
    'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask/fine_mesh/imgs_pt_test',
    # r'Plenoxels ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/syn/ckpt_eval_cuvol/thresh_10/ckpt/imgs_pt_test',
    # r'MipNeRF360 ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/multinerf/results/blender/{}/pts/lv_10_pts/imgs_pt_test',
    # 'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12/ckpt_eval_surf_masked/ckpt/imgs_pt_no_color',
    'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12/ckpt_eval_surf_masked/ckpt/imgs_pt',
    
    # 'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse_norm_l12': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse_norm_l12_no_color': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12/ckpt_eval_surf_masked/ckpt/imgs_pt_no_color',
    # 'solid_less_trunc_no_sparse_norm_l12_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12_2/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse_norm_l12_3': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12_3/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse_norm_l12_high_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12_high_tv/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse_norm_l12_2_high_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12_2_high_tv/ckpt_eval_surf_masked/ckpt/imgs_pt',
    # 'solid_less_trunc_no_sparse_norm_l12_2_high_tv_edge': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc_no_sparse_norm_l12_2_high_tv_edge/ckpt_eval_surf_masked/ckpt/imgs_pt',
}

scenes = {
    # "ficus": 24, 
    # "ficus_400": 24, 
    # "ficus_200": 24,
    # "ficus": 36, 
    # "ficus_400": 36, 
    # "ficus_200": 36,
    "ship_re": 0, 
    "ship_re_400": 0, 
    "ship_re_200": 0,
    # "lyre": 36, 
    # "lyre_400": 36, 
    # "lyre_200": 36,
}


special_index = {
}

special_group=["ficus", "ship_re", "lyre"]
# special_group=[]

special_group_replace={
    # 'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask_white/fine_mesh/imgs_pt_test',
    'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/good_trunc/solid_less_trunc/ckpt_eval_surf_masked/ckpt/imgs_pt_test',
}


scene_name_map = {
    'ficus_200' : r'$200 \times 200$',
    'ficus_400' : r'$400 \times 400$',
    'ficus' : r'$800 \times 800$',
    'ship_re_200' : r'$200 \times 200$',
    'ship_re_400' : r'$400 \times 400$',
    'ship_re' : r'$800 \times 800$',
}

fig,axes = make_plot(scenes, methods, transpose=True,
                     special_index=special_index, scene_name_map=scene_name_map, special_group=special_group, special_group_replace=special_group_replace,
                     fontsize=25, spacing=0.02, verbose=False, subplot_size=(5,2.7), img_size=(500, 250))

# out_path = 'paper/low_res.png'
out_path = 'paper/low_res.pdf'
fig.savefig(out_path, facecolor='white', bbox_inches='tight', dpi=100)

print(f"Saved to {out_path}")
