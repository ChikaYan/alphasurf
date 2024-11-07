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
    
    

    




fontsize=40
verbose=True
subplot_size=(5,5)


nrow = 3
ncols = 4
img_size=(1000,1000)

fig,axes = plt.subplots(
    nrows=nrow, ncols=ncols,
    figsize=(ncols * subplot_size[1],nrow * subplot_size[0]),
    gridspec_kw={'wspace':0.03,'hspace':0.2}
    )


def obtain_kwargs(scene, method=None):
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
                pass
                # kwargs['patch'] = ((300, 510), (130, 130))
                # kwargs['patch_zoom'] = 2.5
                # kwargs['patch_loc'] = 'bl'
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
        if method == 'RGB':
            kwargs['take_patch'] = ((300, 120), (250, 350))
        else:
            kwargs['take_patch'] = ((100, 40), (500, 450))

    elif scene == 'ship_re':
        kwargs['take_patch'] = ((0, 100), (550, 550))
    elif scene.replace('.','') == 'rabit':
        if method == 'RGB':
            kwargs['take_patch'] = ((300, 200), (500, 500))
        else:
            kwargs['take_patch'] = ((300, 350), (500, 240))



    return kwargs



for i in range(nrow):
    for j in range(ncols):
        if i == 0:
            if j == 0:
                img_idx = 0
                scene = 'ship_re'
                method = 'neuralangelo'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/neuralangelo/logs/syn/ship_re/imgs_pt_test"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel(r"Neuralangelo: \textbf{0.28}", fontsize=fontsize)
                ax.set_ylabel('(a)', fontsize=fontsize)

            if j == 1:
                img_idx = 0
                scene = 'ship_re'
                method = 'Ours'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/ship_re/good_trunc/solid_less_trunc/ckpt_eval_surf_masked/ckpt/imgs_pt_test"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel(r"Ours: \textbf{0.28}", fontsize=fontsize)

            if j == 2:
                img_idx = 1
                scene = 'monkey'
                method = 'neuralangelo'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/neuralangelo/logs/syn/monkey/imgs_no_crop"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel(r"Neuralangelo: 3.22", fontsize=fontsize)

            if j == 3:
                img_idx = 1
                scene = 'monkey'
                method = 'Ours'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/monkey/norm_l12/norm_l12_decay_9_low_tv_2/ckpt_eval_surf_masked/ckpt/imgs_no_crop"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel(r"Ours: \textbf{0.78}", fontsize=fontsize)


        if i == 1:
            if j == 0:
                img_idx = 9
                scene = 'floss_small'
                method = 'RGB'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/data/real/{scene}/images"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("RGB", fontsize=fontsize)
                ax.set_ylabel('(b)', fontsize=fontsize)

            if j == 1:
                img_idx = 9
                scene = 'floss_small'
                method = ''
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/Geo-Neus/exp/{scene}/womask_bg/fine_mesh/imgs_pt_train"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("GeoNeuS", fontsize=fontsize)

            if j == 2:
                img_idx = 9
                scene = 'floss_small'
                method = ''
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{scene}/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/imgs_pt_train"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("Plenoxels", fontsize=fontsize)

            if j == 3:
                img_idx = 9
                scene = 'floss_small'
                method = ''
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{scene}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("Ours", fontsize=fontsize)
            

        if i == 2:
            if j == 0:
                img_idx = 5
                scene = 'flower_thin'
                method = 'RGB'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/data/real/{scene}/images"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("RGB", fontsize=fontsize)
                ax.set_ylabel('(c)', fontsize=fontsize)

            if j == 1:
                img_idx = 5
                scene = 'flower_thin'
                method = ''
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{scene}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_no_norm_no_tv/ckpt_eval_surf_masked/imgs_pt_train"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("Ours", fontsize=fontsize)

            ################ rabit ##############
            if j == 2:
                img_idx = 0
                scene = 'rabit'
                method = 'RGB'
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/data/real/{scene}/images"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("RGB", fontsize=fontsize)
                ax.set_ylabel('(d)', fontsize=fontsize)

            if j == 3:
                img_idx = 0
                scene = 'rabit'
                method = ''
                imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{scene}/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/imgs_pt_train"


                kwargs = obtain_kwargs(scene, method)
                image_set = ImageSet(
                    imgs_path.format(scene.replace('.','')), 
                    verbose=verbose, 
                    # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
                    img_size=img_size, 
                    img_ext='*',
                    **kwargs)

                ax = axes[i, j]
                    
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

                ax.set_xlabel("Ours", fontsize=fontsize)
            ################ rabit End ##############
            # ################ cup ##############
            # if j == 2:
            #     img_idx = 99
            #     scene = 'cup_yellow_circle'
            #     method = ''
            #     imgs_path = f"/rds/project/rds-qxpdOeYWi78/Geo-Neus/exp/{scene}/womask_bg/fine_mesh/imgs_pt_train"


            #     kwargs = obtain_kwargs(scene, method)
            #     image_set = ImageSet(
            #         imgs_path.format(scene.replace('.','')), 
            #         verbose=verbose, 
            #         # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
            #         img_size=img_size, 
            #         img_ext='*',
            #         **kwargs)

            #     ax = axes[i, j]
                    
            #     try:
            #         im = image_set[img_idx]
            #         ax.imshow(im)
            #     except Exception as e:
            #         ax.imshow(np.ones([*img_size, 3]))
            #         print(e)


            #     # make spines (the box) invisible
            #     plt.setp(ax.spines.values(), visible=False)
            #     # remove ticks and labels for the left axis
            #     ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

            #     ax.set_xlabel("GeoNeuS", fontsize=fontsize)
            #     ax.set_ylabel('(d)', fontsize=fontsize)

            # if j == 3:
            #     img_idx = 99
            #     scene = 'cup_yellow_circle'
            #     method = ''
            #     imgs_path = f"/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{scene}/llff_mlv_bg/l2_norm_no_sparse_d16_less_trunc_tv_10_zero_lv/ckpt_eval_surf_masked/imgs_pt_train"


            #     kwargs = obtain_kwargs(scene, method)
            #     image_set = ImageSet(
            #         imgs_path.format(scene.replace('.','')), 
            #         verbose=verbose, 
            #         # img_name_pattern = 'r_{}.png' if method=='RGB' else '{:05d}.png', 
            #         img_size=img_size, 
            #         img_ext='*',
            #         **kwargs)

            #     ax = axes[i, j]
                    
            #     try:
            #         im = image_set[img_idx]
            #         ax.imshow(im)
            #     except Exception as e:
            #         ax.imshow(np.ones([*img_size, 3]))
            #         print(e)


            #     # make spines (the box) invisible
            #     plt.setp(ax.spines.values(), visible=False)
            #     # remove ticks and labels for the left axis
            #     ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

            #     ax.set_xlabel("Ours", fontsize=fontsize)
            # ################ cup End ##############









out_path = 'paper/rebuttal2.png'
out_path = 'paper/rebuttal_cup.pdf'
out_path = 'paper/rebuttal_rabbit.pdf'

fig.savefig(out_path, facecolor='white', bbox_inches='tight', dpi=100)

print(f"Saved to {out_path}")
