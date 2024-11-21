#!/bin/bash
#!
#! Example SLURM job script for Peta4-IceLake (Ice Lake CPUs, HDR200 IB)
#! Last updated: Sat Jul 31 15:39:45 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J nerf-surf-cpu
#! Which project should be charged:
#SBATCH -A MANTIUK-SL2-CPU
#! SBATCH -A OZTIRELI-SL3-CPU
#SBATCH -p icelake
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=76
#! How much wallclock time will be required?
#SBATCH --time=04:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by cpu number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*76 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3380 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

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
module load rhel8/default-icl              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

source /usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh
conda activate voxel_sdf

#! Full path to application executable: 
application=""

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="/rds/project/rds-qxpdOeYWi78/plenoxels/opt"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 76:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
#CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
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

#! sbatch --array=0- ./hpc_eval_batch_blender_cpu.sh

CONF_DIR="./configs/hpc_confs"

BATCH_NAME="good_trunc"
# BATCH_NAME="hotdog_debug"

# DATA_DIR="../data/nerf_synthetic/materials"
# DATA_DIR="../data/nerf_synthetic/ficus"
# DATA_DIR="../data/nerf_synthetic/drums"
# DATA_DIR="../data/nerf_synthetic/lego_re"
DATA_DIR="../data/nerf_synthetic/ship_re"
# DATA_DIR="../data/nerf_synthetic/chair_re"
# DATA_DIR="../data/nerf_synthetic/mic_re"
# DATA_DIR="../data/nerf_synthetic/hotdog_re"


# DATA_DIR="../data/nerf_synthetic/lego_transparent"
# DATA_DIR="../data/nerf_synthetic/lotus"
# DATA_DIR="../data/nerf_synthetic/lyre"
# DATA_DIR="../data/nerf_synthetic/maple"
# DATA_DIR="../data/nerf_synthetic/bulb"
# DATA_DIR="../data/nerf_synthetic/ficus_transparent"
# DATA_DIR="../data/nerf_synthetic/crystal"
# DATA_DIR="../data/nerf_synthetic/glasses"
# DATA_DIR="../data/nerf_synthetic/vase"

# DATA_DIR="../data/nerf_synthetic/bee"
# DATA_DIR="../data/nerf_synthetic/stair"
# DATA_DIR="../data/nerf_synthetic/hoop"
# DATA_DIR="../data/nerf_synthetic/fence"
# DATA_DIR="../data/nerf_synthetic/windwheel"
# DATA_DIR="../data/nerf_synthetic/glasses2"
# DATA_DIR="../data/nerf_synthetic/glass_table"
# DATA_DIR="../data/nerf_synthetic/swing"
# DATA_DIR="../data/nerf_synthetic/effiel"
# DATA_DIR="../data/nerf_synthetic/pot"
# DATA_DIR="../data/nerf_synthetic/jar"
# DATA_DIR="../data/nerf_synthetic/bulb"



CONFS=( "$CONF_DIR/$BATCH_NAME/"* )
CONF_NAME=$(basename ${CONFS[$SLURM_ARRAY_TASK_ID]})
EXP_NAME="tuning/$(basename $DATA_DIR)/$BATCH_NAME/$(basename $CONF_NAME .yaml)"
CONFIG="$CONF_DIR/$BATCH_NAME/$CONF_NAME"
CKPT_DIR=ckpt/$EXP_NAME


echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG




OUT_PATH=$CKPT_DIR/ckpt_eval_surf_masked

PT_PATH=$OUT_PATH/ckpt_nc_pts.npy
echo "Saving pts to: " $PT_PATH
python eval_cf_blender.py --input_path $PT_PATH --gt_path $DATA_DIR --out_dir $OUT_PATH/remeshed \
    --run_alpha_shape --alpha_shape_alpha 0.001



OUT_PATH=$CKPT_DIR/ckpt_eval_surf
PT_PATH=$OUT_PATH/ckpt_nc_pts.npy
echo "Saving pts to: " $PT_PATH
python eval_cf_blender.py --input_path $PT_PATH --gt_path $DATA_DIR --out_dir $OUT_PATH/remeshed \
    --run_alpha_shape --alpha_shape_alpha 0.001






#! sbatch --array=0- ./hpc_eval_batch_blender_cpu.sh

