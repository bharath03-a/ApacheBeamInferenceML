#!/bin/bash

set -e

export PROJECT_ID="ccda-centauri-dev"
export REGION="us-central1"
export GCS_BUCKET_NAME="apache-beam-inference-ml-bucket"

echo "--- Using Project: $PROJECT_ID and Region: $REGION ---"

gsutil -m cp ../data/ClinVarVCVRelease_00-latest.xml gs://$GCS_BUCKET_NAME/xml_data/
# gsutil -m cp ../data/variant_summary.txt gs://$GCS_BUCKET_NAME/txt_data/