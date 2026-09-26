#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J v4train
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/v4_train_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
hostname; nproc; free -g | head -2
$PY train_shift.py --work ../work_v3 --frac 0.19 --Kuse 3 --procs 44 --tag shift
