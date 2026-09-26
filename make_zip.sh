#!/bin/bash
# Usage: ./make_zip.sh <output_dir_name e.g. output_v5a> [team_name]
set -e
V=${1:-output_v5a}; TEAM=${2:-TEAM}
ROOT=/u/student/2025/cs25mtech14011/amazon_ml_2026
STAGE=$(mktemp -d /tmp/pkg_XXXX)
mkdir -p $STAGE/output $STAGE/code/business_entity_resolution
cp $ROOT/business_entity_resolution/$V/matching_results.tsv $ROOT/business_entity_resolution/$V/candidate_pairs.tsv $STAGE/output/
rsync -a --exclude __pycache__ $ROOT/business_entity_resolution/src $STAGE/code/business_entity_resolution/
cp $ROOT/business_entity_resolution/README.md $ROOT/business_entity_resolution/requirements.txt $STAGE/code/business_entity_resolution/
cp $ROOT/Documentation_template.md $STAGE/
rm -f $ROOT/${TEAM}_submission.zip
(cd $STAGE && zip -qr $ROOT/${TEAM}_submission.zip output code Documentation_template.md)
rm -rf $STAGE
ls -la $ROOT/${TEAM}_submission.zip; unzip -l $ROOT/${TEAM}_submission.zip | tail -30
