#!/bin/bash
#!
#! Example SLURM job script for Wilkes2 (Broadwell, ConnectX-4, P100)
#! Last updated: Mon 13 Nov 12:06:57 GMT 2017
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J nerf-surf-test
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A OZTIRELI-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=2:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime. 

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

source /usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh
conda activate voxel_sdf

#! Full path to application executable: 
#! application="/rds/user/tw554/hpc-work/workspace/hypernerf/foo.sh"
#! application="/rds/user/tw554/hpc-work/workspace/hypernerf/$TARGET_SCRIPT"
#! application = $app

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="/rds/project/rds-qxpdOeYWi78/plenoxels/opt"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 12:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"


echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

#! sbatch --array=0- ./hpc_train_eval_good_scannerf.sh

CONF_DIR="./configs/hpc_confs"

BATCH_NAME="good_trunc"

SCENES=("helicopter1" "plant3" "bulldozer2")

BACKEND="surf_trav"
NEAR_CLIP=0.


# CONFS=("solid_less_trunc" "single_lv") 
# CONFS=("solid_less_trunc") 
# CONFS=("solid_less_trunc_high_tv" "single_lv") 
CONFS=("solid_less_trunc_high_norm" "solid_less_trunc_high_tv") 
CONFS=("single_lv_high_norm" "single_lv_high_tv") 
CONFS=("solid_less_trunc_high_norm_high_tv" "solid_less_trunc_high_norm_high_tv_2") 
CONFS=("solid_less_trunc_high_norm_high_tv_3") 
CONFS=("solid_less_trunc_high_norm_high_tv_4") 
CONFS=("solid_less_trunc_high_norm_high_tv_5") 


BATCH_NAME="abla_less_trunc"
# # # CONFS=("single_lv_high_norm_high_tv" "single_lv_high_norm_high_tv_2") 
# # CONFS=("single_lv_high_norm_high_tv_3") 
# CONFS=("single_lv_high_norm_high_tv_4") 
CONFS=("single_lv_high_norm_high_tv_5") 
# CONFS=("single_lv") 

SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
DATA_DIR="../data/scannerf/$SCENE"
NERF_NAME="syn"
# NERF_NAME="syn_tv"
# NERF_NAME="syn_tv_2"
for CONF in ${CONFS[@]}
do
    if [[ $CONF == "single_lv" ]]
    then
        BATCH_NAME="abla_less_trunc"
    fi
    if [[ $CONF == "single_lv_high_norm" ]]
    then
        BATCH_NAME="abla_less_trunc"
    fi
    if [[ $CONF == "single_lv_high_tv" ]]
    then
        BATCH_NAME="abla_less_trunc"
    fi
    if [[ $CONF == "single_lv_high_norm_low_tv" ]]
    then
        BATCH_NAME="abla_less_trunc"
    fi

    CONF_NAME=${CONF}.yaml
    EXP_NAME="tuning/$(basename $DATA_DIR)/$BATCH_NAME/$NERF_NAME/$(basename $CONF_NAME .yaml)"
    CONFIG="$CONF_DIR/$BATCH_NAME/$CONF_NAME"
    CKPT_DIR=ckpt/$EXP_NAME

    mkdir -p $CKPT_DIR

    echo CKPT $CKPT_DIR
    echo DATA_DIR $DATA_DIR
    echo CONFIG $CONFIG

    python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG --load_pretrain_density_sh "/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$(basename $DATA_DIR)/nerf/$NERF_NAME"

    python render_imgs_circle.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND --near_clip $NEAR_CLIP

    python render_imgs_circle.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND --render_depth --depth_thresh 0.1 --near_clip $NEAR_CLIP

    # python render_imgs.py $CKPT_DIR $DATA_DIR --renderer_backend $BACKEND --num_views 50 --near_clip $NEAR_CLIP


    CKPT_PATHS=$CKPT_DIR/ckpt*.npz
    OUT_PATH=$CKPT_DIR/ckpt_eval_surf_single
    mkdir -p $OUT_PATH
    for ckpt_path in ${CKPT_PATHS[@]}
    do
        ckpt_name=$(basename $ckpt_path)

        PT_PATH=$OUT_PATH/"$(basename $ckpt_name .npz)"_nc_pts.npy
        echo "Saving pts to: " $PT_PATH

        python exract_points_no_cam.py $CKPT_DIR/$ckpt_name --downsample_density 0.001 \
            --out_path $PT_PATH --intersect_th -1 --n_sample 5 --single_lv

        python eval_cf_blender.py --input_path $PT_PATH --out_dir $OUT_PATH/"$(basename $ckpt_name .npz)"

        VIS_IN=$OUT_PATH/"$(basename $ckpt_name .npz)"/vis_d2s.ply

        # xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
        #     --input_path $VIS_IN --out_dir $OUT_PATH/"$(basename $ckpt_name .npz)"/imgs_pt

        xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh_blender.py \
            --input_path $VIS_IN --out_dir $OUT_PATH/"$(basename $ckpt_name .npz)"/imgs_pt_test --n_imgs -1 \
            --no_color \
            --c2w_path /rds/project/rds-qxpdOeYWi78/plenoxels/data/scannerf/$SCENE/train_100_c2w.npy
    done

done



#! sbatch --array=0- ./hpc_train_eval_good_scannerf.sh

