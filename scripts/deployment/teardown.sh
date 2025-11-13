#!/bin/bash
# Teardown deployment

set -e

PROVIDER=${1:-aws}
CLUSTER_NAME=${2:-llm-cluster}

echo "Tearing down $CLUSTER_NAME on $PROVIDER..."

case $PROVIDER in
    aws)
        eksctl delete cluster --name $CLUSTER_NAME --wait
        ;;
    gcp)
        gcloud container clusters delete $CLUSTER_NAME --quiet
        ;;
    azure)
        az aks delete --name $CLUSTER_NAME --resource-group llm-platform --yes
        ;;
    local)
        docker-compose down -v
        ;;
esac

echo "✓ Teardown complete"
