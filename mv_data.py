from pathlib import Path
import os


scenes = ("coffe_table","kitchen_table","hourglass","monkey","double_table")



for scene in scenes:
    # os.system(f"ls opt/ckpt/tuning/{scene}/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/ckpt/")
    os.system(f"mv opt/ckpt/tuning/{scene}/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/ckpt/* opt/ckpt/tuning/{scene}/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/")

    print()
