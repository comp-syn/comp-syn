#!/bin/bash -e

DOCKER_REPO="ialcloud"
IMAGE_NAME="comp-syn-base"
BROWSERS_BASE_VERSION="1.0.1-py3.8.10"
IMAGE_VERSION="1.0.0"

docker build --build-arg BROWSERS_BASE_VERSION=${BROWSERS_BASE_VERSION} -t ${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_VERSION} .

if [ "${1}" == "--push" ]; then
  docker push ${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_VERSION}
fi
