#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --account=bbsg-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuH200x8
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p slurm_logs

#================ 软件环境 =================
# 先 source conda，再 deactivate（否则会报错）
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda deactivate || true
conda deactivate || true

# 清理并加载 cuda 模块（先卸载可能冲突的 cudatoolkit）
module purge || true
module reset || true
module unload cudatoolkit || true
module load cuda/12.8

conda activate /work/hdd/bdeg/yzhang62/anaconda3/envs/cloud_seg

#================ 分布式环境 ===============
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
master_addr=$(head -n 1 <<< "$nodes")

export MASTER_ADDR="$master_addr"
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
# 若无 IB 或不确定，先禁用：
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com

echo "######################################################################"
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Job Name            : ${SLURM_JOB_NAME}"
echo "Partition           : ${SLURM_JOB_PARTITION}"
echo "Node List           : ${SLURM_JOB_NODELIST}"
echo "GPUs per Node       : ${SLURM_GPUS_ON_NODE}"
echo "MASTER_ADDR         : ${MASTER_ADDR}"
echo "MASTER_PORT         : ${MASTER_PORT}"
echo "######################################################################"
echo ""

nvidia-smi || true
cd /work/hdd/bdeg/yzhang62/cloud_seg 

python train.py --fold 2
python train.py --fold 3
python train.py --fold 4    

echo "========================================="
echo "Job finished with exit code $?."
echo "========================================="

