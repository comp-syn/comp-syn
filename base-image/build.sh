#!/bin/bash -e

DOCKER_REPO="ialcloud"
IMAGE_NAME="compsyn-base-image"
IMAGE_VERSION="0.0.1"

docker build -t ${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_VERSION} .
