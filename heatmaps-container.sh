#!/bin/bash
# Builds and runs a heatmaps container.
#
# Usage:
# ./heatmaps-container.sh \
#      [HEATMAPS OPTIONS]

set -eux

podman build "$(dirname -- "$0")"
image_id=$(podman build -q "$(dirname -- "$0")")

podman run --rm -ti \
    --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    --volume $HOME:$HOME \
    --volume /mnt:/mnt \
    --volume /run/media:/run/media \
    "$image_id" \
    "$@"
