#!/bin/bash
export CONTAINER_NAME="isaac-sim"
docker rm $CONTAINER_NAME

set -e

# login to nvcr.io
echo "Logging in to nvcr.io..."
docker login nvcr.io

# pull docker image
echo "Pulling docker image..."
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# access to X11
# xhost +local:docker 
xhost +
notify-send 'X11 Access Control:' 'Access granted to local Docker containers for GUI applications.'


echo "Setting variables..."
command="$@"
if [[ -z "$@" ]]; then
    command="bash"
fi
# Set to desired Nucleus
omni_server="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2"
if ! [[ -z "${OMNI_SERVER}" ]]; then
	omni_server="${OMNI_SERVER}"
fi
# Set to desired Nucleus username
omni_user="admin"
if ! [[ -z "${OMNI_USER}" ]]; then
	omni_user="${OMNI_USER}"
fi
# Set to desired Nucleus password
omni_password="admin"
if ! [[ -z "${OMNI_PASS}" ]]; then
	omni_password="${OMNI_PASS}"
fi
# Set to "Y" to accept EULA
accept_eula="Y"
if ! [[ -z "${ACCEPT_EULA}" ]]; then
	accept_eula="${ACCEPT_EULA}"
fi
# Set to "Y" to opt-in
privacy_consent="Y"
if ! [[ -z "${PRIVACY_CONSENT}" ]]; then
	privacy_consent="${PRIVACY_CONSENT}"
fi
# Set to an email or unique user name
privacy_userid="${omni_user}"
if ! [[ -z "${PRIVACY_USERID}" ]]; then
	privacy_userid="${PRIVACY_USERID}"
fi

# echo "Logging in to nvcr.io..."
# docker login nvcr.io

# echo "Pulling docker image..."
# docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# echo "Running Isaac Sim container with X11 forwarding..."
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=${accept_eula}" --rm \
	-v $HOME/.Xauthority:/root/.Xauthority \
	-e DISPLAY \
	-e "OMNI_USER=${omni_user}" -e "OMNI_PASS=${omni_password}" \
	-e "OMNI_SERVER=${omni_server}" \
    -e "PRIVACY_CONSENT=${privacy_consent}" -e "PRIVACY_USERID=${privacy_userid}" \
    -v ~/docker/isaac-sim/kit/cache/Kit:/isaac-sim/kit/cache:rw \
	-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
	-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
	-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
	-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
	-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
	-v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
	-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
	-v ~/docker/isaac-sim/documents:/root/Documents:rw \
	nvcr.io/nvidia/isaac-sim:4.2.0 \
	-c "${command}"

echo "Isaac Sim container run completed!"
