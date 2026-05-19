#!/bin/bash

# deploy.sh - Deploy para Google Cloud Run

set -e

PROJECT_ID=$1
SERVICE_NAME=reddit-mlops
REGION=us-central1

# Autenticar no GCP
echo "Autenticando no GCP..."
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Build da imagem (sem enviar source)
echo "Building Docker image..."
gcloud builds submit \
  --no-source \
  --config=cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=$SERVICE_NAME,_PROJECT_ID=$PROJECT_ID \
  --project=$PROJECT_ID

# Deploy no Cloud Run
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