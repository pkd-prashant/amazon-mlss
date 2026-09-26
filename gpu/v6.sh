#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J v6
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/v6_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
$PY v6.py --work ../work_v3 --out ../output_v6 --procs 44
$PY cmp_outputs_v6.py
cd /u/student/2025/cs25mtech14011/my_data/student_resource && $PY utils/validate_submission.py --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v6/matching_results.tsv --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v6/candidate_pairs.tsv --test-dir dataset/test | tail -1
