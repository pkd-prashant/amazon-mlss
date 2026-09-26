#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J v8
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/v8_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
$PY v8.py --work ../work_v3 --out ../output_v8 --alpha 0.5 --procs 44
cd /u/student/2025/cs25mtech14011/my_data/student_resource && $PY utils/validate_submission.py --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v8/matching_results.tsv --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v8/candidate_pairs.tsv --test-dir dataset/test | tail -1
