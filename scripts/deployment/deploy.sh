#!/bin/bash
# One-click deployment script for Distributed LLM Platform
# Supports AWS, GCP, and Azure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Distributed LLM Platform Deployment"
echo "=================================="
echo ""

# Parse arguments
PROVIDER=${1:-aws}
CLUSTER_NAME=${2:-llm-cluster}
NUM_NODES=${3:-2}
GPU_TYPE=${4:-g4dn.xlarge}
REGION=${5:-us-west-2}

echo "Configuration:"
echo "  Provider: $PROVIDER"
echo "  Cluster: $CLUSTER_NAME"
echo "  Nodes: $NUM_NODES"
echo "  GPU Type: $GPU_TYPE"
echo "  Region: $REGION"
echo ""

# Confirm
read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

# Deploy based on provider
case $PROVIDER in
    aws)
        echo "${GREEN}Deploying to AWS...${NC}"
        ./scripts/deployment/deploy_aws.sh "$CLUSTER_NAME" "$NUM_NODES" "$GPU_TYPE" "$REGION"
        ;;
    gcp)
        echo "${GREEN}Deploying to GCP...${NC}"
        ./scripts/deployment/deploy_gcp.sh "$CLUSTER_NAME" "$NUM_NODES" "$GPU_TYPE" "$REGION"
        ;;
    azure)
        echo "${GREEN}Deploying to Azure...${NC}"
        ./scripts/deployment/deploy_azure.sh "$CLUSTER_NAME" "$NUM_NODES" "$GPU_TYPE" "$REGION"
        ;;
    local)
        echo "${GREEN}Deploying locally with Docker Compose...${NC}"
        docker-compose up -d
        ;;
    *)
        echo "${RED}Unknown provider: $PROVIDER${NC}"
        echo "Supported: aws, gcp, azure, local"
        exit 1
        ;;
esac

echo ""
echo "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Access your services:"
echo "  Ray Dashboard: http://<cluster-ip>:8265"
echo "  MLflow: http://<cluster-ip>:5000"
echo "  Model API: http://<cluster-ip>:8000"
echo ""
echo "Next steps:"
echo "  1. Configure kubectl: kubectl config use-context $CLUSTER_NAME"
echo "  2. Check pods: kubectl get pods -n ray-llm"
echo "  3. Run training: llm-train --model gpt2 --dataset wikitext"
