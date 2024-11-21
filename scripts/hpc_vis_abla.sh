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
#SBATCH -J vis_pt_mesh
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
#SBATCH --time=05:00:00
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

#! sbatch ./hpc_vis_abla.sh

CONF_DIR="./configs/hpc_confs"

BATCH_NAME="good_trunc"

SCENES=("ship_re" "drums" "ficus" "materials" "lego_re" "chair_re" "mic_re" "hotdog_re" \
       "lyre" "bee" "stair" "fence" "scale" "seat" "well" "slide" \
       "glass_table" "vase"  "jar" "glasses2" "skull" "perfume" "wine" "bottle" ) # 0~23
SCENES=("lego_transparent" "bottle_2" "stool" "cup" "cupboard" "case") 


SCENES=("ship_re" "drums" "ficus" "materials" "lego_re" "chair_re" "mic_re" "hotdog_re" \
       "lyre" "bee" "stair" "fence" "scale" "seat" "well" "slide")
# SCENES=("glass_table" "jar" "glasses2" "wine" "lego_transparent" "bottle_2" "cup" "case") 
# SCENES=("bottle_2") 


# SCENES=("ship_re" "drums" "ficus" "materials" "lego_re" "chair_re" "mic_re" "hotdog_re" \
#        "lyre" "bee" "stair" "fence" "scale" "seat" "well" "slide")
SCENES=("lego_re" "ship_re" "ficus" "lyre" "stair")
# SCENES=("glass_table" "case")
SCENES=("ficus")

for SCENE in ${SCENES[@]}
do
    IN_PATHS=()
    #  masked surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_conv/ckpt_eval_surf_masked/ckpt/vis_d2s.ply")
    #  single surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_conv/ckpt_eval_surf_single/ckpt/vis_d2s.ply")
    
    #  masked surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_trunc/ckpt_eval_surf_masked/ckpt/vis_d2s.ply")
    #  single surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_trunc/ckpt_eval_surf_single/ckpt/vis_d2s.ply")
    
    #  masked surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_tv_no_norm/ckpt_eval_surf_masked/ckpt/vis_d2s.ply")
    #  single surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_tv_no_norm/ckpt_eval_surf_single/ckpt/vis_d2s.ply")
    
    #  masked surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_tv/ckpt_eval_surf_masked/ckpt/vis_d2s.ply")
    #  single surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_tv/ckpt_eval_surf_single/ckpt/vis_d2s.ply")
    
    #  masked surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_conv/ckpt_eval_surf_masked/ckpt/vis_d2s.ply")
    #  single surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_no_conv/ckpt_eval_surf_single/ckpt/vis_d2s.ply")
    
    #  masked surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_single_lv/ckpt_eval_surf_masked/ckpt/vis_d2s.ply")
    #  single surf
    IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/solid2_single_lv/ckpt_eval_surf_single/ckpt/vis_d2s.ply")


    
    # #  masked surf
    # IN_PATH1="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/trans_no_trunc/ckpt_eval_surf_masked/ckpt/vis_d2s.ply"
    # #  single surf
    # IN_PATH2="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/trans_no_trunc/ckpt_eval_surf_single/ckpt/vis_d2s.ply"
    
    # #  masked surf
    # IN_PATH1="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/trans_no_tv_no_norm/ckpt_eval_surf_masked/ckpt/vis_d2s.ply"
    # #  single surf
    # IN_PATH2="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/trans_no_tv_no_norm/ckpt_eval_surf_single/ckpt/vis_d2s.ply"
    
    # #  masked surf
    # IN_PATH1="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/trans_no_tv/ckpt_eval_surf_masked/ckpt/vis_d2s.ply"
    # #  single surf
    # IN_PATH2="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/abla/trans_no_tv/ckpt_eval_surf_single/ckpt/vis_d2s.ply"

    
    for IN_PATH in ${IN_PATHS[@]}
    do
        # OUT_DIR="$(dirname $IN_PATH)/imgs_pt"
        # echo Visualizing: $IN_PATH

        # xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
        #     --input_path $IN_PATH --out_dir $OUT_DIR --extra_ele -30

        OUT_DIR="$(dirname $IN_PATH)/imgs_pt_crop"
        echo Visualizing: $IN_PATH
        xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
            --input_path $IN_PATH --out_dir $OUT_DIR --mask_crop
    done
    
    # OUT_DIR="$(dirname $IN_PATH1)/imgs_pt"
    # echo Visualizing: $IN_PATH1

    # xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
    #     --input_path $IN_PATH1 --out_dir $OUT_DIR

    # OUT_DIR="$(dirname $IN_PATH2)/imgs_pt"
    # echo Visualizing: $IN_PATH2

    # xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
    #     --input_path $IN_PATH2 --out_dir $OUT_DIR

    
    # OUT_DIR="$(dirname $IN_PATH1)/imgs_pt_crop"
    # echo Visualizing: $IN_PATH1

    # xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
    #     --input_path $IN_PATH1 --out_dir $OUT_DIR --mask_crop

    # OUT_DIR="$(dirname $IN_PATH2)/imgs_pt_crop"
    # echo Visualizing: $IN_PATH2

    # xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh.py \
    #     --input_path $IN_PATH2 --out_dir $OUT_DIR --mask_crop


done



#! sbatch ./hpc_vis_abla.sh

