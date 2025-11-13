"""
Model Serving CLI Interface

Command-line interface for deploying models with Ray Serve.

Usage:
    llm-serve --model-path ./outputs/my-model --port 8000
"""

import argparse
import logging
import sys
import time

from src.serving.server import ModelServer, DeploymentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed LLM Platform - Model Serving CLI"
    )

    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--name", type=str, default="llm-model", help="Deployment name"
    )
    parser.add_argument(
        "--num-replicas", type=int, default=2, help="Number of replicas"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="GPUs per replica"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    return parser.parse_args()


def main():
    """Main entry point for serving CLI."""
    args = parse_args()

    logger.info("Deploying model with Ray Serve")

    # Create deployment config
    config = DeploymentConfig(
        model_path=args.model_path,
        name=args.name,
        num_replicas=args.num_replicas,
        ray_actor_options={"num_gpus": args.num_gpus},
    )

    # Deploy model
    server = ModelServer(config)
    server.deploy()

    logger.info(f"Model deployed successfully!")
    logger.info(f"Server running on http://localhost:{args.port}")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server.undeploy()
        logger.info("Server stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
