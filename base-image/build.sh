#!/bin/bash -e

DOCKER_REPO="ialcloud"
IMAGE_NAME="compsyn-base"
BROWSERS_BASE_VERSION="1.0.0"
IMAGE_VERSION="1.0.0"

docker build --build-arg BROWSERS_BASE_VERSION=1.0.0 -t ${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_VERSION} .
