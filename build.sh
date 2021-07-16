#!/bin/bash

PROJECT_ID=$(gcloud config get-value core/project)

IMAGE_NAME='finetune4textgen-gpu'
IMAGE_TAG='latest'
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG

docker build . -f ./images/gcp/gpu/Dockerfile -t $IMAGE_URI
docker push $IMAGE_URI

echo $IMAGE_URI
