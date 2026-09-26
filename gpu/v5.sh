#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J v5
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 05:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/v5_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
PY=/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3
$PY v5.py --work ../work_v3 --knn_dir ../../gpu --out_root .. --extra 2 --procs 44
for v in v5a v5b; do
  cd /u/student/2025/cs25mtech14011/my_data/student_resource
  $PY utils/validate_submission.py --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_$v/matching_results.tsv --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_$v/candidate_pairs.tsv --test-dir dataset/test | tail -2 | sed "s/^/[$v] /"
done
