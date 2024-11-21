

#### Input the Scene Name here ####
SCENE='ship'


cd ./opt

DATA_DIR="../data/$SCENE"

#### First, train plenoxels ####
CONF_DIR="./configs/plenoxels"
CONF_NAME=syn.yaml
CONFIG="$CONF_DIR/$CONF_NAME"

BACKEND="cuvol"

EXP_NAME="$(basename $DATA_DIR)/plenoxels/$(basename $CONF_NAME .yaml)"
CKPT_DIR=ckpt/$EXP_NAME
NERF_DIR=$CKPT_DIR

mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG



#### Then train our method ####
CONF_DIR="./configs"
CONF_NAME="syn.yaml"

EXP_NAME="$(basename $DATA_DIR)/$(basename $CONF_NAME .yaml)"
CONFIG="$CONF_DIR/$CONF_NAME"
CKPT_DIR=ckpt/$EXP_NAME
mkdir -p $CKPT_DIR
BACKEND="surf_trav"

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG --load_pretrain_density_sh "$NERF_DIR"


#### (Optional) Render in Circle ####
python render_imgs_circle.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND

python render_imgs_circle.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND --render_depth --depth_thresh 0.1



#### Point Cloud Extraction and Evaluation ####
OUT_PATH=$CKPT_DIR/points
mkdir -p $OUT_PATH
ckpt_name=ckpt.npz

PT_PATH=$OUT_PATH/pts.npy
echo "Saving pts to: " $PT_PATH

python exract_points_no_cam.py $CKPT_DIR/$ckpt_name --downsample_density 0.001 \
    --out_path $PT_PATH --intersect_th 0.1 --n_sample 5

python eval_cf_blender.py --input_path $PT_PATH --gt_path $DATA_DIR --out_dir $OUT_PATH



