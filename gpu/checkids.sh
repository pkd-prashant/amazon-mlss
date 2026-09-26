#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J checkids
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/gpu/checkids_%j.out
cd /u/student/2025/cs25mtech14011/my_data/student_resource
for v in output_v5a output_v5c; do
  echo "== $v"
  /u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3 utils/validate_submission.py --check-ids \
    --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/$v/matching_results.tsv \
    --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/$v/candidate_pairs.tsv \
    --test-dir dataset/test | tail -3
done
