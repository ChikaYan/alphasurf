#/bin/bash


EXP_NAME="more_ups/surf_con"
CONFIG="./configs/surface_cuda_syn.yaml"

SCAN=40
# SCAN=110
DATA_DIR="../data/dtu/dtu_scan$SCAN"
DTU_CF_EVAL=true

BACKEND="surf_trav"


CKPT_DIR=ckpt/$EXP_NAME


mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

python render_imgs_circle.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND

python render_imgs_circle.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND --render_depth --depth_thresh 0.1


DTU_DIR="/rds/project/rds-qxpdOeYWi78/plenoxels/data/dtu_eval/SampleSet/MVS Data"
OUT_PATH=$CKPT_DIR/ckpt_eval
mkdir -p $OUT_PATH
ckpt_name="ckpt.npz"
PT_PATH=$OUT_PATH/"$(basename $ckpt_name .npz)"_pts.npy

python exract_points.py $CKPT_DIR/$ckpt_name $DATA_DIR --traj_type "test" --downsample_density 0.2 \
    --out_path $PT_PATH --renderer_backend $BACKEND --intersect_th 0.1

python eval_dtu.py --pts_dir $PT_PATH --scan $SCAN --dataset_dir "$DTU_DIR" --out_dir $OUT_PATH/"$(basename $ckpt_name .npz)"

