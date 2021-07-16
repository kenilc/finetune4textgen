#!/bin/bash

PROJECT_ID=$(gcloud config get-value core/project)

# Your ML service account's secret_key.json
GOOGLE_APPLICATION_CREDENTIALS=

# Your data files and training jobs must be in the same region
REGION=us-central1

# Jobs must have unique names for each run
JOB_NAME="XXXXX_`date +%Y%m%d_%H%M%S`"

BUCKET_NAME=

# Storage for installed packages, logs and model checkpoints
JOB_DIR="gs://${BUCKET_NAME}/${JOB_NAME}"

# Training and evaluation files stored on GCS
TRAIN_CSV_FILES=
EVAL_CSV_FILES=

# The docker image created by ./build.sh
IMAGE_URI=gcr.io/${PROJECT_ID}/finetune4textgen-gpu:latest

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier custom \
    --master-machine-type n1-standard-8 \
    --master-accelerator type=nvidia-tesla-t4,count=1 \
    --region ${REGION} \
    --job-dir ${JOB_DIR} \
    -- \
    --disable_tqdm True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --learning_rate 1e-5 \
    --warmup_steps 1000 \
    --num_train_epochs 10 \
    --log_level info \
    --logging_dir ./logs \
    --logging_strategy steps \
    --logging_steps 5000 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 20000 \
    --group_by_length True \
    --length_column_name length \
    --do_train True \
    --do_eval True \
    --train_csv_files ${TRAIN_CSV_FILES} \
    --eval_csv_files ${EVAL_CSV_FILES} \
    --output_dir ./outputs
