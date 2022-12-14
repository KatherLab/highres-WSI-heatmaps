#!/bin/bash
# Builds and runs a heatmaps container.
#
# Usage:
# ./heatmaps-container.sh \
#      [CONTAINER OPTIONS]
#      -- \
#      [HEATMAPS OPTIONS]
#
# i.e. a -- is used to separate the podman options from heatmaps options

set -e

podman_args=()
while [[ $# -gt  0 ]]; do
    case $1 in
        --) shift; break;; # we're done with podman args
        *) podman_args+=("$1"); shift;; # append to podman args
    esac
done

image_id=$(podman build -q .)

podman run --rm -ti \
    --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    "${podman_args[@]}" \
    "$image_id" \
    "$@"
