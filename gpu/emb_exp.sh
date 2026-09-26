#!/bin/bash
#SBATCH -p cse-gpu-all
#SBATCH -J embexp
#SBATCH --gres=gpu:V100-SXM3-32GB:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/gpu/emb_exp_%j.out
export HF_HOME=/tmp/$USER/hf; mkdir -p $HF_HOME
/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3 /u/student/2025/cs25mtech14011/amazon_ml_2026/gpu/emb_exp.py
