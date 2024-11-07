import numpy as np
import configargparse
from pathlib import Path
import sklearn.neighbors as skln
import os

parser = configargparse.ArgumentParser()


parser.add_argument(
    "--scene",
    type=str,
)

parser.add_argument(
    "--method",
    type=str,
    default='plenoxels'
)
parser.add_argument(
    "--downsample_density",
    type=float,
    default=0.001,
)


args = parser.parse_args()

if args.method == 'plenoxels':
    npy_dir = Path(f'/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/{args.scene}/nerf/syn/ckpt_eval_cuvol/')

    npy_paths = [
        npy_dir / 'thresh_10' / 'ckpt_nc_pts.npy',
        npy_dir / 'thresh_30' / 'ckpt_nc_pts.npy',
        npy_dir / 'thresh_50' / 'ckpt_nc_pts.npy',
        npy_dir / 'thresh_70' / 'ckpt_nc_pts.npy',
        npy_dir / 'thresh_90' / 'ckpt_nc_pts.npy',
    ]

    out_dir = npy_dir / 'thresh_10,30,50,70,90' / 'ckpt_nc_pts.npy'
    out_dir.parent.mkdir(exist_ok=True, parents=True)


elif args.method == 'mipnerf360':
    npy_dir = Path(f'/rds/project/rds-qxpdOeYWi78/multinerf/results/blender/{args.scene}/pts/')

    npy_paths = [
        npy_dir / 'lv_10_pts.npy',
        npy_dir / 'lv_30_pts.npy',
        npy_dir / 'lv_50_pts.npy',
        npy_dir / 'lv_70_pts.npy',
        npy_dir / 'lv_90_pts.npy',
    ]

    out_dir = npy_dir / 'lv_10,30,50,70,90_pts.npy'




# all_pts = []

# for npy_path in npy_paths:
#     pts = np.load(str(npy_path))
#     all_pts.append(pts)

# all_pts = np.concatenate(all_pts, axis=0)

# # run downsample 
# nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=args.downsample_density, algorithm='kd_tree', n_jobs=-1)
# nn_engine.fit(all_pts)
# rnn_idxs = nn_engine.radius_neighbors(all_pts, radius=args.downsample_density, return_distance=False)
# mask = np.ones(all_pts.shape[0], dtype=np.bool_)
# for curr, idxs in enumerate(rnn_idxs):
#     if mask[curr]:
#         mask[idxs] = 0
#         mask[curr] = 1
# all_pts = all_pts[mask]

# np.save(str(out_dir), all_pts)



# os.system(f"python eval_cf_blender.py --input_path {str(out_dir)} --gt_path ../data/nerf_synthetic/{args.scene} --out_dir {str(out_dir.parent)}")

os.system(f"xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh_blender.py \
            --input_path {str(out_dir.parent / 'vis_d2s.ply')} --out_dir {str(out_dir.parent / 'imgs_pt_test_crop')} --n_imgs 60 --mask_crop")


