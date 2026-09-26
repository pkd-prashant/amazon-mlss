#!/bin/bash
#SBATCH -p cse-gpu-all
#SBATCH -J cetest
#SBATCH --gres=gpu:V100-SXM3-32GB:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/ce_test_%j.log
export HF_HOME=/tmp/$USER/hf; mkdir -p $HF_HOME
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3 cross_encoder.py ../emb test
