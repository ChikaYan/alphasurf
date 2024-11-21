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
#SBATCH --time=00:30:00
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

#! sbatch --array=0- ./hpc_vis_llff.sh

SCENES=("trex" "leaves" "flower_thin")
SCENES=("glass_cup" "rough")
SCENES=("plastic" "fern" "orchids")
SCENES=("floss_small" "heart")
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

IN_PATHS=()

# # ours single surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/llff/single_lv_less_trunc/ckpt_eval_surf_single/ckpt_nc_pts.ply")
# # ours single surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/llff/slv_fast_3/ckpt_eval_surf_single/ckpt_nc_pts.ply")

# Ours
IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/llff_mlv_bg/l2_norm_no_sparse_d2_less_trunc_tv_10_zero_lv_high_n_2/ckpt_eval_surf_masked/ckpt_nc_pts.ply")



# # nerf threshold 50
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/synllff/ckpt_eval_cuvol/thresh_50/ckpt_nc_pts.ply")
# # nerf threshold 10
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/synllff/ckpt_eval_cuvol/thresh_10/ckpt_nc_pts.ply")
IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/synllff_bg/ckpt_eval_cuvol/thresh_10/ckpt_nc_pts.ply")

################ transparent ################
# # ours single surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/llff_low_tv/multi_lv/ckpt_eval_surf_single/ckpt_nc_pts.ply")
# # ours single surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/llff_low_tv/multi_lv_large_tv/ckpt_eval_surf_single/ckpt_nc_pts.ply")


# # nerf threshold 50
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/synllff_low_tv/ckpt_eval_cuvol/thresh_50/ckpt_nc_pts.ply")
# # nerf threshold 10
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/synllff_low_tv/ckpt_eval_cuvol/thresh_10/ckpt_nc_pts.ply")



# # NeuS
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/NeuS/exp/$SCENE/womask/fine_mesh/vis_d2s.ply")


for IN_PATH in ${IN_PATHS[@]}
do
    OUT_DIR="$(dirname $IN_PATH)/imgs_pt_train"
    echo Visualizing: $IN_PATH

    xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh_llff.py \
        --input_path $IN_PATH --out_dir $OUT_DIR --dataset "/rds/project/rds-qxpdOeYWi78/plenoxels/data/real/$SCENE"


done







#! sbatch --array=0- ./hpc_vis_llff.sh

