#!/usr/bin/env bash 
# Sbatch settings
#SBATCH --partition cpu_tp
#SBATCH --exclusive
#SBATCH --qos 8nodespu
# Standard output
#SBATCH --nodes 2
#SBATCH -o output/%x.out
# Standard error
#SBATCH -e  output/%x.err
# time

module purge
# module load openmpi/4.1.5/gcc-12.3.0
export STARPU_FXT_TRACE=1
export STARPU_FXT_PREFIX=~/disk_tmp_mpi
export STARPU_SCHED=dmda 

echo "=========== Job Information =========="
echo "Node List : "$SLURM_NODELIST
echo "my jobID : "$SLURM_JOB_ID
echo " Partition : " $SLURM_JOB_PARTITION
echo " submit directory : " $SLURM_SUBMIT_DIR
echo " submit host : " $SLURM_SUBMIT_HOST
echo " In the directory : " $PWD
echo "As the user : " $USER
echo "=========== Job Information =========="
#mkdir -p output

nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n " "${nodelist[@]}" > output/nodefile
mpirun -N 5  --hostfile output/nodefile  ./build/gemm 
rm output/nodefile
