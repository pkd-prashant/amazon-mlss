#!/bin/bash
#SBATCH -p cse-gpu-all
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 00:10:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/gpu/test_%j.out
hostname
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
source /u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/activate 2>/dev/null
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
$PY -c "import torch,sentence_transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'st',sentence_transformers.__version__)"
timeout 20 curl -sI https://huggingface.co | head -1 || echo "no internet"
