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

import imageio
import cv2
from tqdm import tqdm
parser = configargparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--traj_type',
                    choices=['spiral', 'circle', 'front'],
                    default='spiral',
                    help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
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
	"--num_views", "-N", type=int, default=200,
    help="Number of frames to render"
)

# Path adjustment
parser.add_argument(
    "--offset", type=str, default="0,0,0", help="Center point to rotate around (only if not --traj)"
)
parser.add_argument("--radius", type=float, default=2.5, help="Radius of orbit (only if not --traj)")
parser.add_argument(
    "--elevation",
    type=float,
    default=-45.0,
    help="Elevation of orbit in deg, negative is above",
)
parser.add_argument(
    "--elevation2",
    type=float,
    default=-12.0,
    help="Max elevation, only for spiral",
)
parser.add_argument(
    "--vec_up",
    type=str,
    default=None,
    help="up axis for camera views (only if not --traj);"
    "3 floats separated by ','; if not given automatically determined",
)
parser.add_argument(
    "--vert_shift",
    type=float,
    default=0.0,
    help="vertical shift by up axis"
)

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")


# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
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

if args.vec_up is None:
    up_rot = dset.c2w[:, :3, :3].cpu().numpy()
    ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
    args.vec_up = np.mean(ups, axis=0)
    args.vec_up /= np.linalg.norm(args.vec_up)
    print('  Auto vec_up', args.vec_up)
else:
    args.vec_up = np.array(list(map(float, args.vec_up.split(","))))

# args.vec_up = args.vec_up @ np.array([[-1, 0, 0],
#                                       [ 0,-1, 0],
#                                       [ 0, 0, 1]])

args.offset = np.array(list(map(float, args.offset.split(","))))
# args.traj_type = 'front'
# args.traj_type = 'circle'
if args.traj_type == 'spiral':
    # angles = np.linspace(-180, 180, args.num_views + 1)[:-1]
    # elevations = np.linspace(args.elevation, args.elevation2, args.num_views)
    # c2ws = [
    #     pose_spherical(
    #         angle,
    #         ele,
    #         args.radius,
    #         args.offset,
    #         vec_up=args.vec_up,
    #     )
    #     for ele, angle in zip(elevations, angles)
    # ]
    # c2ws += [
    #     pose_spherical(
    #         angle,
    #         ele,
    #         args.radius,
    #         args.offset,
    #         vec_up=args.vec_up,
    #     )
    #     for ele, angle in zip(reversed(elevations), angles)
    # ]

    repeats = 10

    angles = np.linspace(-180, 180, (args.num_views) // repeats + 1)[:-1]
    angles = np.concatenate([angles for _ in range(repeats)])
    elevations = np.linspace(-90, 90, args.num_views)
    c2ws = [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(elevations, angles)
    ]
elif args.traj_type == 'front':
    args.vec_up = args.vec_up @ np.array([[-1, 0, 0],
                                        [ 0,-1, 0],
                                        [ 0, 0, 1]])

    c2ws = [
        pose_spherical(
            angle,
            args.elevation,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ]

    # positions = dset.c2w[:, :3, 3].cpu().numpy()
    # center = np.average(positions, axis=0)
    # center_axis = (0. - center) / np.linalg.norm(center)
    # axis_2 = center_axis.copy()
    # axis_2[0] += 1
    # axis_a = np.cross(center_axis, axis_2)
    # axis_a = axis_a / np.linalg.norm(axis_a)
    # axis_b = np.cross(center_axis, axis_a)
    # axis_b = axis_b / np.linalg.norm(axis_b)

    # # radius ratio decides the circle radius of test camera
    # RADIUS_RATIO = 0.5
    # N_TEST = 100

    # dists = []
    # for i in range(positions.shape[0]):
    #     dists.append(np.sqrt(np.sum((positions[i,...] - center) ** 2)))
    # mean_dist = np.average(dists)
    # radius = mean_dist * RADIUS_RATIO
    # print(f'radius: {radius}')

    # angles = np.linspace(0, 2 * np.pi, N_TEST)
    # for i, ang in enumerate(angles):  
    #     new_pos = center + radius * np.cos(ang) * axis_a + radius * np.sin(ang) * axis_b

    #     test_cam = camera_model.Camera.from_json(str(camera_gt_dir / rgb_list[0].stem) + '.json')
    #     test_cam.position = new_pos
    #     test_cam.look_at(new_pos, np.array([0,0,0]), np.array([1,0,0]))

    #     with open(str(camera_test_gt_dir / f'{i:06d}.json'), 'w') as f:
    #         json.dump(test_cam.to_json(), f, indent=2)
else :
    c2ws = [
        pose_spherical(
            angle,
            args.elevation,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ]
c2ws = np.stack(c2ws, axis=0)
if args.vert_shift != 0.0:
    c2ws[:, :3, 3] += np.array(args.vec_up) * args.vert_shift
c2ws = torch.from_numpy(c2ws).to(device=device)

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

np.save(path.join(path.dirname(args.ckpt), 'train_cams.npy'), dset.c2w)
np.save(path.join(path.dirname(args.ckpt), 'test_cams.npy'), c2ws.cpu().numpy())

render_out_path = path.join(path.dirname(args.ckpt), 'circle_renders')

if args.render_depth:
    render_out_path += '_depth'

# Handle various image transforms
if args.crop != 1.0:
    render_out_path += f'_crop{args.crop}'
if args.vert_shift != 0.0:
    render_out_path += f'_vshift{args.vert_shift}'

grid = svox2.SparseGrid.load(args.ckpt, device=device)
# args.renderer_backend = grid.surface_type
print(grid.center, grid.radius)

# grid.density_data.data = torch.clamp_min(grid.density_data.data, torch.logit(torch.tensor(0.1)).to(grid.density_data.device))

# DEBUG
#  grid.background_data.data[:, 32:, -1] = 0.0
#  render_out_path += '_front'

if grid.use_background:
    if args.nobg:
        grid.background_data.data[..., -1] = 0.0
        render_out_path += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_out_path += '_nofg'

    #  # DEBUG
    #  grid.background_data.data[..., -1] = 100.0
    #  a1 = torch.linspace(0, 1, grid.background_data.size(0) // 2, dtype=torch.float32, device=device)[:, None]
    #  a2 = torch.linspace(1, 0, (grid.background_data.size(0) - 1) // 2 + 1, dtype=torch.float32, device=device)[:, None]
    #  a = torch.cat([a1, a2], dim=0)
    #  c = torch.stack([a, 1-a, torch.zeros_like(a)], dim=-1)
    #  grid.background_data.data[..., :-1] = c
    #  render_out_path += "_gradient"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_out_path += '_blackbg'
    grid.opt.background_brightness = 0.0

render_out_path += '.mp4'
print('Writing to', render_out_path)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = c2ws.size(0)
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    frames = []
    #  if args.near_clip >= 0.0:
    grid.opt.near_clip = 0.0 #args.near_clip
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


