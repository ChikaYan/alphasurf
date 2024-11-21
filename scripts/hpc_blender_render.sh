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
#SBATCH -J blender
#! Which project should be charged:
#SBATCH -A MANTIUK-SL2-CPU
#! SBATCH -A OZTIRELI-SL3-CPU
#SBATCH -p icelake
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=8
#! How much wallclock time will be required?
#SBATCH --time=4:00:00
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
workdir="/rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
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

BLENDER_PATH=/rds/project/rds-qxpdOeYWi78/plenoxels/blender/blender

SCENE_NAME="lotus"
SCENE_NAME="maple"
SCENE_NAME="lamp"
# SCENE_NAME="lyre"
SCENE_NAME="coffee"
# SCENE_NAME="dice"
SCENE_NAME="bulb"
SCENE_NAME="crystal"
SCENE_NAME="ficus_transparent"
SCENE_NAME="glasses"
SCENE_NAME="vase"
SCENE_NAME="ship_re"
# SCENE_NAME="lego_re"
SCENE_NAME="chair_re"
# SCENE_NAME="hotdog_re"
# SCENE_NAME="mic_re"

SCENE_NAME="stair"
SCENE_NAME="bee"
SCENE_NAME="hoop"
SCENE_NAME="windwheel"
SCENE_NAME="fence"
SCENE_NAME="glasses2"
SCENE_NAME="glass_table"
SCENE_NAME="swing"
# SCENE_NAME="effiel"
# SCENE_NAME="pot"
# SCENE_NAME="jar"

SCENE_NAMES=("scale" "seat" "well" "house" "swing")
SCENE_NAMES=("skull" "perfume" "wine")
SCENE_NAMES=("bottle")
# SCENE_NAMES=("lego_transparent")
SCENE_NAMES=("swing2" "round" "slide" "bridge")
SCENE_NAMES=("bottle_2" "stool" "cup" "cupboard")
SCENE_NAMES=("case")
SCENE_NAMES=("lego_re" "ficus" "ship_re")

SCENE_NAMES=("ship_re" "drums" "ficus" "materials" "lego_re" "chair_re" "mic_re" "hotdog_re" \
       "lyre" "bee" "stair" "fence" "scale" "seat" "well" "slide" \
       "glass_table" "vase"  "jar" "glasses2" "skull" "perfume" "wine" "bottle" \
       "lego_transparent" "bottle_2" "stool" "cup" "cupboard" "case") # ~29

SCENE_NAMES=("case")

SCENE_NAMES=(candle coffe_table kitchen_table bulb_new)
SCENE_NAMES=(car hourglass ship_bottle)
SCENE_NAMES=(monkey)

SCENE_NAMES=(burger chest window)
SCENE_NAMES=(pot_burger)
SCENE_NAMES=(double_table)
SCENE_NAMES=(bottle_only)
SCENE_NAMES=(leaf_vase toothbrush dinning_table door)

SCENE_NAME=${SCENE_NAMES[$SLURM_ARRAY_TASK_ID]}

SCENE_PATH="/rds/project/rds-qxpdOeYWi78/plenoxels/data/nerf_synthetic/blender/$SCENE_NAME.blend"


$BLENDER_PATH $SCENE_PATH --background --python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/blender_render_train.py -- $SCENE_NAME 

python exr_to_pts.py $SCENE_NAME

$BLENDER_PATH $SCENE_PATH --background --python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/blender_render_test.py -- $SCENE_NAME 


# $BLENDER_PATH $SCENE_PATH --background --python /rds/project/rds-qxpdOeYWi78/plenoxels/opt/scripts/blender_render_demo.py -- $SCENE_NAME



#! sbatch --array=0- hpc_blender_render.sh