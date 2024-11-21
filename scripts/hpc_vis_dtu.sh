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

#! sbatch --array=0- ./hpc_vis_dtu.sh

SCANS=(63 24 37 69) # ~3
SCANS=(63 24 83 105 37 40 55 65 69 97 106 110 114 118 122) #~14
SCANS=(37 40 63 69 110) # ~4
SCAN=${SCANS[$SLURM_ARRAY_TASK_ID]}
SCENE="dtu_scan$SCAN"

IN_PATHS=()

# # ours masked surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/dtu_new_exp/less_trunc/ckpt_eval_surf_masked/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # ours single surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/dtu_new_exp/less_trunc/ckpt_eval_surf_single/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # ours mlv surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/dtu_new_exp/less_trunc/ckpt_eval_surf_mlv/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")

# # nerf depth mode
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # nerf threshold 100
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn/ckpt_eval_cuvol/thresh_100/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # nerf threshold 50
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn/ckpt_eval_cuvol/thresh_50/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # nerf threshold 10
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn/ckpt_eval_cuvol/thresh_10/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")

# # NeuS
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/NeuS/dtu_exp/$SCENE/wmask/fine_mesh/vis_$(printf "%03d" $SCAN)_d2s.ply")

# # HFS
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/HFS/exps_1.5/$SCENE/womask_hfs/fine_mesh/vis_d2s.ply")


###### low scale ######
# # ours masked surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/dtu_ls/smooth/ckpt_eval_surf_masked/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # ours single surf
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/dtu_ls/smooth/ckpt_eval_surf_single/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")

# # nerf depth mode
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn_lowscale/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # nerf threshold 50
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn_lowscale/ckpt_eval_cuvol/thresh_50/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")
# # nerf threshold 10
# IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn_lowscale/ckpt_eval_cuvol/thresh_10/ckpt/vis_$(printf "%03d" $SCAN)_d2s.ply")

# NeuS
IN_PATHS+=("/rds/project/rds-qxpdOeYWi78/NeuS/dtu_exp/$SCENE/wmask/fine_mesh/vis_$(printf "%03d" $SCAN)_d2s.ply")


for IN_PATH in ${IN_PATHS[@]}
do
    OUT_DIR="$(dirname $IN_PATH)/imgs_pt_test_crop"
    echo Visualizing: $IN_PATH

    xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh_dtu.py \
        --input_path $IN_PATH --out_dir $OUT_DIR --scan $SCAN --mask_crop


done


# # GT 
# IN_PATH="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/ckpt/tuning/$SCENE/nerf/syn/ckpt_eval_cuvol/depth_spiral_mode_0.1/remeshed/vis_$(printf "%03d" $SCAN)_s2d.ply"
# OUT_DIR="$(dirname $IN_PATH)/imgs_gt_test_crop"
# echo Visualizing: $IN_PATH

# xvfb-run -a python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/vis_pt_mesh_dtu.py \
#     --input_path $IN_PATH --out_dir $OUT_DIR --scan $SCAN --no_color --mask_crop





#! sbatch --array=0- ./hpc_vis_dtu.sh

