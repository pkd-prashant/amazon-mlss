#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J v9
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/v9_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
$PY v9.py --work ../work_v3 --emb ../emb --out ../output_v9 --procs 44 && $PY hybrid_coverage.py ../work_v3 ../output_v9 ../output_v7 ../output_v9h
cd /u/student/2025/cs25mtech14011/my_data/student_resource && $PY utils/validate_submission.py --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v9h/matching_results.tsv --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v9h/candidate_pairs.tsv --test-dir dataset/test | tail -1
