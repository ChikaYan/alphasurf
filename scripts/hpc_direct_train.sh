
cd /rds/project/rds-qxpdOeYWi78/plenoxels/opt

CONF_DIR="./configs/hpc_confs"
BATCH_NAME="udf_alpha_lap"
CONFS="opt/configs/hpc_confs/udf_alpha_lap/mid_large_var.yaml"
# DATA_DIR="../data/nerf_synthetic/materials"
DATA_DIR="../data/nerf_synthetic/lego"
# DATA_DIR="../data/dtu/dtu_scan63"

CONF_NAME=$(basename $CONFS)
EXP_NAME="tuning/$(basename $DATA_DIR)/$BATCH_NAME/$(basename $CONF_NAME .yaml)"
CONFIG="$CONF_DIR/$BATCH_NAME/$CONF_NAME"
CKPT_DIR=ckpt/$EXP_NAME

mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

python render_imgs_circle.py $CKPT_DIR $DATA_DIR
