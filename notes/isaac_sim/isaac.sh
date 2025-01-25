#!/bin/bash

export CONTAINER_NAME="isaac-sim"
docker rm $CONTAINER_NAME
docker login nvcr.io
docker pull nvcr.io/nvidia/isaac-sim:4.2.0
xhost +local:docker && notify-send 'X11 Access Control:' 'Access granted to local Docker containers for GUI applications.'

docker run --name $CONTAINER_NAME --entrypoint bash -it --runtime=nvidia --gpus all \
-e "PRIVACY_CONSENT=Y" \
-e "ACCEPT_EULA=Y" \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
nvcr.io/nvidia/isaac-sim:4.2.0