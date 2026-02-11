#!/bin/bash
# Common functions for mll_cluster run_embodiment scripts

# Get the first task ID for a given config
# libero_10 (long) starts from task 3, all others start from task 0
get_first_task_id() {
    local config_name="$1"
    local config_tag=$(extract_config_tag "$config_name")
    if [ "$config_tag" = "long" ]; then
        echo "3"
    else
        echo "0"
    fi
}

# Extract config tag from CONFIG_NAME
# If config ends with _openvlaoft or _eval, don't set CONFIG_TAG
# Otherwise, extract the part after the last _
extract_config_tag() {
    local config_name="$1"
    local config_tag=""
    
    if [[ ! "$config_name" =~ _openvlaoft$ ]] && [[ ! "$config_name" =~ _eval$ ]]; then
        if [[ "$config_name" =~ _([^_/]+)$ ]]; then
            config_tag="${BASH_REMATCH[1]}"
        fi
    fi
    echo "$config_tag"
}

# Inject config tag into log directory path
# Replaces "logs/" with "logs_${CONFIG_TAG}/" in the path
# Handles both relative paths (./logs/...) and absolute paths
inject_config_tag_into_log_path() {
    local log_dir="$1"
    local config_tag="$2"
    
    if [ -n "$config_tag" ]; then
        log_dir=$(echo "$log_dir" | sed "s|/logs/|/logs_${config_tag}/|" | sed "s|^logs/|logs_${config_tag}/|")
    fi
    echo "$log_dir"
}

# Derive eval config name from training config name
# Pattern: insert _eval before any tag suffix (e.g., _cam2 becomes _eval_cam2)
# Base: libero_spatial_grpo_openvlaoft -> libero_spatial_grpo_openvlaoft_eval
# With tag: libero_spatial_grpo_openvlaoft_cam2 -> libero_spatial_grpo_openvlaoft_eval_cam2
derive_eval_config_name() {
    local train_config="$1"
    local eval_config="$train_config"
    
    # If config doesn't already end with _eval, insert _eval before any tag
    if [[ ! "$train_config" =~ _eval$ ]] && [[ ! "$train_config" =~ _eval_ ]]; then
        # Check if config ends with _openvlaoft (base case)
        if [[ "$train_config" =~ _openvlaoft$ ]]; then
            eval_config="${train_config}_eval"
        # Check if config ends with _openvlaoft_{tag} (tagged case)
        elif [[ "$train_config" =~ _openvlaoft_([^/]+)$ ]]; then
            # Extract the tag part (e.g., "cam2" from "libero_spatial_grpo_openvlaoft_cam2")
            local tag="${BASH_REMATCH[1]}"
            # Replace _openvlaoft_{tag} with _openvlaoft_eval_{tag}
            eval_config=$(echo "$train_config" | sed "s|_openvlaoft_${tag}$|_openvlaoft_eval_${tag}|")
        fi
    fi
    echo "$eval_config"
}
