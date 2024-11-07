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
        
    def draw_box(self, img, left, right, top, bottom, line_width=5, line_color=(1,0,0)):
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
            left = max((width - new_width)//2, 0)
            top = max((height - new_height)//2,0)
            right = min((width + new_width)//2, width)
            bottom = min((height + new_height)//2, height)
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
              subplot_size=(5, 3.2), fontsize=16, spacing=0.1, img_size=(1000,1000), verbose=False):
    
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

            if scene.replace('.','') in special_group_replace:
                replace = special_group_replace[scene.replace('.','')]
                if method in replace:
                    imgs_path = replace[method]

            img_idx = scenes[scene]
            if method in special_index and scene in special_index[method]:
                img_idx = special_index[method][scene]


            # if scene == 'cup_yellow_circle.':
            #     print()

            # scene = scene.replace('.','')


            kwargs = {}            
            kwargs['zoom'] = 1.5
            if method == 'RGB':
                kwargs['zoom'] *= 0.25

            if scene == 'glass_cup':
                # kwargs['take_patch'] = np.array(((140, 140), (300, 300)))
                if method == 'RGB':
                    kwargs['take_patch'] = ((280, 300), (600, 600))
                else:
                    kwargs['take_patch'] = ((370, 370), (400, 330))
                pass
            elif scene == 'rough':
                kwargs['take_patch'] = ((140, 140), (250, 250))
            elif scene.replace('.','') == 'cup_yellow_circle':
                if img_idx == 0:
                    if method == 'RGB':
                        kwargs['take_patch'] = ((350, 120), (500, 500))
                    else:
                        kwargs['take_patch'] = ((390, 380), (320, 150))
                elif img_idx == 99:
                    pass
                    if method == 'RGB':
                        kwargs['take_patch'] = ((250, 280), (350, 400))
                    else:
                        kwargs['take_patch'] = ((0, 320), (650, 300))
                        # kwargs['take_patch'] = ((180, 130), (270, 140))
            elif scene.replace('.','') == 'cup_dark_circle':
                if img_idx == 47:
                    if method == 'RGB':
                        kwargs['take_patch'] = ((340, 100), (450, 500))
                    else:
                        kwargs['take_patch'] = ((0, 100), (800, 600))
                        
            elif scene.replace('.','') == 'flower_thin':
                if img_idx == 5:
                    if method == 'RGB':
                        pass
                    else:
                        kwargs['patch'] = ((300, 510), (130, 130))
                        kwargs['patch_zoom'] = 2.5
                        kwargs['patch_loc'] = 'bl'
            #### old size ones ###
            # elif scene.replace('.','') == 'floss':
            #     if img_idx == 10:
            #         if method == 'RGB':
            #             kwargs['take_patch'] = ((310, 150), (250, 400))
            #         else:
            #             kwargs['take_patch'] = ((150, 100), (400, 500))
            #             kwargs['patch'] = ((350, 400), (150, 150))
            #             kwargs['patch_zoom'] = 2
            #             kwargs['patch_loc'] = 'ur'
            # elif scene.replace('.','') == 'floss_small':
            #     if img_idx == 11:
            #         if method == 'RGB':
            #             kwargs['take_patch'] = ((250, 350), (250, 300))
            #         else:
            #             kwargs['take_patch'] = ((150, 250), (400, 400))
            #             # kwargs['patch'] = ((350, 400), (150, 150))
            #             # kwargs['patch_zoom'] = 2
            #             # kwargs['patch_loc'] = 'ur'
            elif scene.replace('.','') == 'floss':
                if img_idx == 10:
                    if method == 'RGB':
                        kwargs['take_patch'] = ((310, 150), (250, 400))
                    else:
                        kwargs['take_patch'] = ((150, 100), (400, 500))
                        kwargs['patch'] = ((350, 400), (150, 150))
                        kwargs['patch_zoom'] = 2
                        kwargs['patch_loc'] = 'ur'
            elif scene.replace('.','') == 'floss_small':
                if img_idx == 10:
                    if method == 'RGB':
                        kwargs['take_patch'] = ((250, 200), (250, 300))
                    else:
                        kwargs['take_patch'] = ((0, 50), (500, 450))
                if img_idx == 9:
                    if method == 'RGB':
                        kwargs['take_patch'] = ((300, 120), (250, 350))
                    else:
                        kwargs['take_patch'] = ((100, 40), (500, 450))
                        # kwargs['patch'] = ((350, 400), (150, 150))
                        # kwargs['patch_zoom'] = 2
                        # kwargs['patch_loc'] = 'ur'
            elif scene.replace('.','') == 'rabit':
                if method == 'RGB':
                    kwargs['take_patch'] = ((300, 200), (500, 500))
                else:
                    kwargs['take_patch'] = ((300, 350), (500, 240))
            elif scene.replace('.','') == 'heart':
                # if method == 'RGB':
                #     kwargs['take_patch'] = ((350, 480), (230, 230))
                # else:
                #     kwargs['take_patch'] = ((350, 450), (250, 250))

                if method == 'RGB':
                    kwargs['take_patch'] = ((350, 480), (230, 230))
                else:
                    kwargs['take_patch'] = ((360, 480), (200, 200))


                
            
            image_set = ImageSet(
                imgs_path.format(scene.replace('.','')), 
                verbose=verbose, 
                # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                img_size=img_size, 
                img_ext='*',
                **kwargs)


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


            scene_name = scene.replace('.', '')

            if not transpose:
                if j == 0:
                    scene_name = scene_name_map[scene_name] if scene_name in scene_name_map else scene_name
                    # scene_name.replace(scene_name[0], scene_name[0].upper(), 1)
                    ax.set_ylabel(scene_name, fontsize=fontsize)
                    
                if i == len(scenes) - 1:
                    ax.set_xlabel(method, fontsize=fontsize)
            else:
                if j == 0:
                    scene_name = scene_name_map[scene] if scene_name in scene_name_map else scene_name
                    # scene_name.replace(scene_name[0], scene_name[0].upper(), 1)
                    ax.set_title(scene_name, fontsize=fontsize)
                    
                if i == 0:
                    ax.set_ylabel(method, fontsize=fontsize)

    return fig, axes



methods = {
    'RGB': '/rds/project/rds-qxpdOeYWi78/plenoxels/data/real/{}/images',
    'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask/fine_mesh/imgs_pt_train',
    'GeoNeuS': '/rds/project/rds-qxpdOeYWi78/Geo-Neus/exp/{}/womask_bg/fine_mesh/imgs_pt_train',
    'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_2_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    
    # 'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/fast_l12_norm_dilate_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'Plenoxels (low tv)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_low_tv_2/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    # r'Plenoxels No Bg ($\sigma=10$)': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_no_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
    
    # 'l2_norm_no_sparse_d16_less_trunc_tv_7_zero_lv_no_tv_ful': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d16_less_trunc_tv_7_zero_lv_no_tv_ful/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d16_less_trunc_tv_12_zero_lv_no_tv_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d16_less_trunc_tv_12_zero_lv_no_tv_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_3': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_3/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_l12_norm_dilate_4_mimic': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_l12_norm_dilate_4_mimic/ckpt_eval_surf_masked/imgs_pt_train',
    # 'mid_l12_norm_dilate_4_mimic_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv/mid_l12_norm_dilate_4_mimic_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'bg/mid_2_l12_norm_dilate_4_mimic': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/mid_2_l12_norm_dilate_4_mimic/ckpt_eval_surf_masked/imgs_pt_train',


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
    
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_no_norm_no_tv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_no_norm_no_tv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_no_norm': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_no_norm/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_less_norm': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_less_norm/ckpt_eval_surf_masked/imgs_pt_train',
    
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_3': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_3/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_4': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_4/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_5': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_5/ckpt_eval_surf_masked/imgs_pt_train',
    # 'l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_6': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_5/ckpt_eval_surf_masked/imgs_pt_train',
    

}

scenes = {
    # "glass_cup": 5,
    # # "rough": 6,
    # # "cup_yellow_circle": 0, 

    "floss_small": 9,
    "cup_yellow_circle.": 99, 
    # "cup_dark_circle": 47,
    # "heart": 16,
    "heart": 32,

    # "flower_thin": 5,
    # "flower_thin.": 23,
    # "trex": 0,
    # "floss": 10,
    # "stand_rl": 0,
    # "rabit": 0,
    # "butterfly": 0,
    # "toy_glass": 0,
    # "hair": 0,
    # "comb": 0,
    # "frog": 0,
    # "rabbit_2": 0,
    # "butterfly_2": 0,
    # "stand_2": 0,
    # "rabbit_3": 0,



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
    'rough': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/fast_l2_norm_surftv_no_sparse_d16_small_tv_3/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'cup_yellow_circle': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d16_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'cup_dark_circle': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'flower_thin': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_no_norm_no_tv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'trex': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'floss': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'floss_small': {
        'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask_bg/fine_mesh/imgs_pt_train',
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'toy_glass': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'stand_rl': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'hair': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'comb': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'rabit': {
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train',
    },
    'heart': {
        # 'NeuS': '/rds/project/rds-qxpdOeYWi78/NeuS/exp/{}/womask_bg/fine_mesh/imgs_pt_train',
        r'Plenoxels': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train',
        'Ours': '/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train',
    },
    
}

scene_name_map = {
    'flower_thin': 'gypso',
    'glass_cup': 'mug',
    'floss_small': 'floss',
    # 'chair_re': 'chair',
    # 'hotdog_re': 'hotdog',
    'rough': 'cup',
    'cup_yellow_circle': 'yellow cup',
    'cup_dark_circle': 'dark cup',
}

fig,axes = make_plot(scenes, methods, transpose=False,
                     special_index=special_index, scene_name_map=scene_name_map,
                     special_group=special_group, special_group_replace=special_group_replace,
                     fontsize=35, spacing=0.02, verbose=True)


# out_path = 'paper/llff_rebuttal.png'
out_path = 'paper/llff.png'
out_path = 'paper/llff.pdf'
fig.savefig(out_path, facecolor='white', bbox_inches='tight', dpi=100)

print(f"Saved to {out_path}")
