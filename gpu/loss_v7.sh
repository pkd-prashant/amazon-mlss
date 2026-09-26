#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J lossv7
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/loss_v7_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
PYTHONPATH=. $PY dev/loss_v7.py
