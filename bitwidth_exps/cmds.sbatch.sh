#!/bin/bash 
# gres_test.sh script 
#SBATCH --gres=gpu:2           # grab 2 gpus 
#SBATCH --cpus-per-task=4      # grab 2 gpus 
#SBACTH --ntasks=18            # going to run 2 tasks 
#SBATCH --job-name=activations # name of job 
#SBATCH --time=4:00:00         # set a time limit (e.g. 4 hours) for the job before it times-out (good practice!) 
#SBATCH --output=slurm.out   # print output to file
# print parameters 

h=`hostname` 
echo "$h $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_TASKID="$SLURM_ARRAY_TASK_ID

srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=32 --bit_a=30 --output=./train_logs_a30_w32/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=30 --bit_a=30 --output=./train_logs_a30_w30/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=28 --bit_a=30 --output=./train_logs_a30_w28/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=26 --bit_a=30 --output=./train_logs_a30_w26/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=24 --bit_a=30 --output=./train_logs_a30_w24/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=22 --bit_a=30 --output=./train_logs_a30_w22/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=20 --bit_a=30 --output=./train_logs_a30_w20/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=18 --bit_a=30 --output=./train_logs_a30_w18/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=16 --bit_a=30 --output=./train_logs_a30_w16/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=32 --bit_a=28 --output=./train_logs_a28_w32/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=30 --bit_a=28 --output=./train_logs_a28_w30/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=28 --bit_a=28 --output=./train_logs_a28_w28/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=26 --bit_a=28 --output=./train_logs_a28_w26/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=24 --bit_a=28 --output=./train_logs_a28_w24/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=22 --bit_a=28 --output=./train_logs_a28_w22/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=20 --bit_a=28 --output=./train_logs_a28_w20/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=18 --bit_a=28 --output=./train_logs_a28_w18/ &
srun --cpus-per-task=4 --gres=gpu:1 rsr-bitwidth-exps.py --bit_w=16 --bit_a=28 --output=./train_logs_a28_w16/ &

wait
