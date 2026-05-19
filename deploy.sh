#!/bin/bash

set -e

PROJECT_ID=$1
SERVICE_NAME=reddit-mlops
REGION=us-central1

echo "Autenticando no GCP..."
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud auth configure-docker

echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

echo "Pushing to Container Registry..."
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI,DAGSHUB_USERNAME=$DAGSHUB_USERNAME,DAGSHUB_TOKEN=$DAGSHUB_TOKEN,CHAMPION_MODEL=$CHAMPION_MODEL \
  --project=$PROJECT_ID

echo "Deploy concluído!"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --project=$PROJECT_ID