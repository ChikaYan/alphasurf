# Copyright 2021 Alex Yu
# Render 360 circle path

import torch
import svox2
import svox2.utils
import math
import configargparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, pose_spherical
from util import config_util
# from skimage.metrics import structural_similarity as ssim_func

import imageio
import cv2
from tqdm import tqdm
from pathlib import Path
parser = configargparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--fps',
                    type=int,
                    default=2,
                    help="FPS of video")
parser.add_argument(
                "--width", "-W", type=float, default=None, help="Rendering image width (only if not --traj)"
                        )
parser.add_argument(
                    "--height", "-H", type=float, default=None, help="Rendering image height (only if not --traj)"
                            )
parser.add_argument(
	"--num_views", "-N", type=int, default=100000,
    help="Number of frames to render"
)

# Path adjustment
parser.add_argument(
    "--offset", type=str, default="0,0,0", help="Center point to rotate around (only if not --traj)"
)


# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")


# Random debugging features
parser.add_argument('--render_depth',
                    action='store_true',
                    default=False,
                    help="Render depth images instead")
parser.add_argument('--depth_thresh',
                    type=float,
                    default=None,
                    help="Sigma/alpha threshold for rendering depth. None means median depth")

args = parser.parse_args()
USE_KERNEL = not args.nokernel
# USE_KERNEL = False
# config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'


dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))



if args.num_views >= dset.c2w.shape[0]:
    c2ws = dset.c2w.numpy()[:, :4, :4]
else:
    test_cam_ids = np.round(np.linspace(0, dset.c2w.shape[0] - 1, args.num_views)).astype(int)
    # test_cam_ids = np.array([48])
    print(f'Using test views with ids: {test_cam_ids}')
    c2ws = dset.c2w.numpy()[test_cam_ids, :4, :4]


c2ws = np.stack(c2ws, axis=0)
c2ws = torch.from_numpy(c2ws).to(device=device)

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')


render_out_path = path.join(path.dirname(args.ckpt), 'test_render_vid')
img_out_path = path.join(path.dirname(args.ckpt), 'test_renders')


if args.render_depth:
    render_out_path += '_depth'

# Handle various image transforms
if args.crop != 1.0:
    render_out_path += f'_crop{args.crop}'

grid = svox2.SparseGrid.load(args.ckpt, device=device)
# args.renderer_backend = grid.surface_type
print(grid.center, grid.radius)

# grid.density_data.data = torch.clamp_min(grid.density_data.data, torch.logit(torch.tensor(0.1)).to(grid.density_data.device))



config_util.setup_render_opts(grid.opt, args)

if args.truncated_vol_render:
    grid.truncated_vol_render_a = 2
    img_out_path = img_out_path + '_trunc'
    print(f"use truncated volume rendering with {grid.truncated_vol_render_a} layers")

os.makedirs(img_out_path, exist_ok=True)

render_out_path += '.mp4'
print('Writing to', render_out_path)

print('Render options', grid.opt)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)

psnrs = []
ssims = []

with torch.no_grad():
    n_images = c2ws.size(0)
    img_eval_interval = 1
    n_images_gen = 0
    frames = []
    #  if args.near_clip >= 0.0:
    # grid.opt.near_clip = 0.0 #args.near_clip
    if args.width is None:
        args.width = dset.get_image_size(0)[1]
    if args.height is None:
        args.height = dset.get_image_size(0)[0]

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = args.height, args.width
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', 0),
                           dset.intrins.get('fy', 0),
                           w * 0.5,
                           h * 0.5,
                           w, h,
                           ndc_coeffs=(-1.0, -1.0))
        torch.cuda.synchronize()
        if args.render_depth:
            im = grid.volume_render_depth_image(cam, args.depth_thresh)
        else:
            im = grid.volume_render_image(cam, use_kernel=USE_KERNEL)
        torch.cuda.synchronize()

        rgb_gt_test = dset.gt[img_id].to(device=device)
        mse = ((rgb_gt_test - im) ** 2).cpu().mean().numpy()
        psnr = -10.0 * math.log10(mse)
        psnrs.append(psnr)
        ssim = compute_ssim(im, rgb_gt_test).item()
        ssims.append(ssim)


        # imageio.imwrite(os.path.join(img_out_path, f"gt_{img_id:04d}.png"), (rgb_gt_test.cpu().numpy() * 255).astype(np.uint8))


        
        if args.render_depth:
            im = viridis_cmap(im.cpu())
        else:
            im.clamp_(0.0, 1.0)
            im = im.cpu().numpy()

        im = (im * 255).astype(np.uint8)
        frames.append(im)
        im = None
        n_images_gen += 1
    if len(frames):
        vid_path = render_out_path
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg

        for i in range(len(frames)):
            imageio.imwrite(os.path.join(img_out_path, f"{i:04d}.png"), frames[i])

all_psnr = np.average(psnrs)
all_ssim = np.average(ssims)

print('PSNR:')
print(psnrs)
print('SSIM:')
print(ssims)

exp_dir = Path(args.ckpt).parent

if args.truncated_vol_render:
    with (exp_dir / 'psnr_truc.txt').open('w') as f:
        f.write(str(all_psnr))
    with (exp_dir / 'ssim_truc.txt').open('w')as f:
        f.write(str(all_ssim))
else:
    with (exp_dir / 'psnr.txt').open('w') as f:
        f.write(str(all_psnr))
    with (exp_dir / 'ssim.txt').open('w')as f:
        f.write(str(all_ssim))
            


