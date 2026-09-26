#!/bin/bash
#SBATCH -p cse-cpu-all
#SBATCH -J sibprior
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH -o /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/work_v3/sibprior_%j.log
cd /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/src
/u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3 sibling_prior.py
cd /u/student/2025/cs25mtech14011/my_data/student_resource && /u/student/2025/cs25mtech14011/CLG-CBM-main/clg_env/bin/python3 utils/validate_submission.py --matching /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v5c/matching_results.tsv --candidate /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v5c/candidate_pairs.tsv --test-dir dataset/test | tail -1
