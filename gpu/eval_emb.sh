#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J evalemb
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/eval_emb_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3 eval_emb.py
