#!/bin/bash

set -e

export PROJECT_ID="ccda-centauri-dev"
export REGION="us-central1"
export GCS_BUCKET_NAME="apache-beam-inference-ml-bucket"
export GCS_DATAFLOW_BUCKET_NAME="apache-beam-inference-ml-dataflow-bucket"
export BQ_DATASET="apache_beam_inference_ml_dataset"

echo "--- Using Project: $PROJECT_ID and Region: $REGION ---"

# infrastructure setup
gcloud storage buckets create "gs://${GCS_BUCKET_NAME}" --location="$REGION" --uniform-bucket-level-access || echo "Bucket already exists."
gcloud storage buckets create "gs://${GCS_DATAFLOW_BUCKET_NAME}" --location="$REGION" --uniform-bucket-level-access || echo "Bucket already exists."
bq --location=$REGION ls --datasets | grep -w "$BQ_DATASET" || bq --location=$REGION mk --dataset "$PROJECT_ID:$BQ_DATASET"

echo "--- Infrastructure setup complete ---"