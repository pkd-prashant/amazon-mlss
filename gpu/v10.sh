#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J v10
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 03:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/v10_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
$PY v10.py --work ../work_v3 --emb ../emb --out ../output_v10 --procs 14
cd /u/student/2025/cs25mtech14011/my_data/student_resource && $PY utils/validate_submission.py --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v10/matching_results.tsv --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v10/candidate_pairs.tsv --test-dir dataset/test | tail -1
