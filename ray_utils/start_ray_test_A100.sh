#!/bin/bash

# Parameter check
if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable not set!"
    exit 1
fi

# Configuration file path (modify according to actual needs)
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
RAY_HEAD_IP_FILE=$REPO_PATH/ray_utils/ray_head_ip.txt
RAY_PORT=${MASTER_PORT:-29500}  # Default port for Ray, can be modified if needed

# Ensure Ray temp directory is user-writable under the repo, not /tmp
# if [ -z "$RAY_TMPDIR" ]; then
#     # USER_NAME=${USER:-$(whoami)}
#     # export RAY_TMPDIR="$REPO_PATH/.ray_tmp/"
#     export RAY_TMPDIR="$REPO_PATH/.tmp/"
# fi
# mkdir -p "$RAY_TMPDIR"
# export TMPDIR="$RAY_TMPDIR"

# Ray object store memory (bytes)
# Can be set via config -> exported as RAY_OBJECT_STORE_MEMORY env var
# Default: 461708984320 (~430GB) for backward compatibility
RAY_MEMORY="${RAY_OBJECT_STORE_MEMORY:-461708984320}"

# Head node startup logic
if [ "$RANK" -eq 0 ]; then
    # Get local machine IP address (assumed to be intranet IP)
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
    # Start Ray head node
    echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
    echo "Ray object store memory: $RAY_MEMORY bytes"
    ray start --head --memory=$RAY_MEMORY --port=$RAY_PORT # --temp-dir="$RAY_TMPDIR"
    
    # Write IP to file
    echo "$IP_ADDRESS" > $RAY_HEAD_IP_FILE
    echo "Head node IP written to $RAY_HEAD_IP_FILE"
else
    # Worker node startup logic
    echo "Waiting for head node IP file..."
    
    # Wait for file to appear (wait up to 360 seconds)
    for i in {1..360}; do
        if [ -f $RAY_HEAD_IP_FILE ]; then
            HEAD_ADDRESS=$(cat $RAY_HEAD_IP_FILE)
            if [ -n "$HEAD_ADDRESS" ]; then
                break
            fi
        fi
        sleep 1
    done
    
    if [ -z "$HEAD_ADDRESS" ]; then
        echo "Error: Could not get head node address from $RAY_HEAD_IP_FILE"
        exit 1
    fi
    
    echo "Starting Ray worker node connecting to head at $HEAD_ADDRESS"
    echo "Ray object store memory: $RAY_MEMORY bytes"
    ray start --memory=$RAY_MEMORY --address="$HEAD_ADDRESS:$RAY_PORT" 
fi
