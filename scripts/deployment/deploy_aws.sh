#!/bin/bash
# Deploy to AWS EKS

set -e

CLUSTER_NAME=$1
NUM_NODES=$2
GPU_TYPE=$3
REGION=$4

echo "Creating EKS cluster..."

# Create EKS cluster
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $REGION \
  --nodegroup-name gpu-nodes \
  --node-type $GPU_TYPE \
  --nodes $NUM_NODES \
  --nodes-min 1 \
  --nodes-max $((NUM_NODES * 2)) \
  --managed \
  --asg-access

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml

# Deploy platform
kubectl apply -f configs/kubernetes/

echo "✓ AWS deployment complete"
