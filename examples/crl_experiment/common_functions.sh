#!/bin/bash
# Common functions for crl_experiment run_embodiment scripts

# Get the first task ID for a given config
get_first_task_id() {
    local config_name="$1"
    local config_tag=$(extract_config_tag "$config_name")
    
    # Check if config is for libero_object
    if [[ "$config_tag" == "object" ]]; then
        echo "5"
    elif [ "$config_tag" = "long" ]; then
        echo "3"
    else
        echo "0"
    fi
}

# Extract config tag from CONFIG_NAME
# If config ends with _openvlaoft, _eval, or _lr, don't set CONFIG_TAG
# Otherwise, extract the part after the last _
# Special handling: if config ends with _lr_{tag}, extract {tag} instead
extract_config_tag() {
    local config_name="$1"
    local config_tag=""
    
    if [[ ! "$config_name" =~ _openvlaoft$ ]] && [[ ! "$config_name" =~ _eval$ ]]; then
        # Check if config ends with _lr_{tag} (e.g., _lr_long)
        if [[ "$config_name" =~ _lr_([^_/]+)$ ]]; then
            # Extract the tag part after _lr_ (e.g., "long" from "_lr_long")
            config_tag="${BASH_REMATCH[1]}"
        # Check if config ends with just _lr (no tag)
        elif [[ ! "$config_name" =~ _lr$ ]]; then
            # Extract the last part after _ (but not if it's _lr)
            if [[ "$config_name" =~ _([^_/]+)$ ]]; then
                config_tag="${BASH_REMATCH[1]}"
            fi
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
# Base: libero_spatial_grpo_openvlaoft_spatial -> libero_spatial_grpo_openvlaoft_eval_spatial
# With tag: libero_spatial_grpo_openvlaoft_cam2 -> libero_spatial_grpo_openvlaoft_eval_cam2
# With lr: libero_spatial_grpo_openvlaoft_lr -> libero_spatial_grpo_openvlaoft_eval_lr
# With lr and tag: libero_10_grpo_openvlaoft_lr_long -> libero_10_grpo_openvlaoft_eval_lr_long
derive_eval_config_name() {
    local train_config="$1"
    local eval_config="$train_config"

    # OPD train configs (..._opd_openvlaoft_<suite>) use the GRPO eval yaml (no separate *_opd_*eval* file).
    # e.g. libero_spatial_opd_openvlaoft_spatial -> libero_spatial_grpo_openvlaoft_eval_spatial
    if [[ "$train_config" =~ _opd_openvlaoft_ ]]; then
        echo "$train_config" | sed 's|_opd_openvlaoft_|_grpo_openvlaoft_eval_|'
        return
    fi

    # If config doesn't already end with _eval, insert _eval before any tag
    if [[ ! "$train_config" =~ _eval$ ]] && [[ ! "$train_config" =~ _eval_ ]]; then
        # Check if config ends with _openvlaoft_lr_{tag} (lr with tag case, e.g., _lr_long)
        if [[ "$train_config" =~ _openvlaoft_lr_([^/]+)$ ]]; then
            # Extract the tag part (e.g., "long" from "libero_10_grpo_openvlaoft_lr_long")
            local tag="${BASH_REMATCH[1]}"
            # Replace _openvlaoft_lr_{tag} with _openvlaoft_eval_lr_{tag}
            eval_config=$(echo "$train_config" | sed "s|_openvlaoft_lr_${tag}$|_openvlaoft_eval_lr_${tag}|")
        # Check if config ends with _openvlaoft_lr (lr without tag case)
        elif [[ "$train_config" =~ _openvlaoft_lr$ ]]; then
            eval_config=$(echo "$train_config" | sed "s|_openvlaoft_lr$|_openvlaoft_eval_lr|")
        # Check if config ends with _openvlaoft (base case)
        elif [[ "$train_config" =~ _openvlaoft$ ]]; then
            eval_config="${train_config}_eval"
        # Check if config ends with _openvlaoft_{tag} (tagged case without lr)
        elif [[ "$train_config" =~ _openvlaoft_([^/]+)$ ]]; then
            # Extract the tag part (e.g., "spatial" from "libero_spatial_grpo_openvlaoft_spatial", "cam2" from "libero_spatial_grpo_openvlaoft_cam2")
            local tag="${BASH_REMATCH[1]}"
            # Replace _openvlaoft_{tag} with _openvlaoft_eval_{tag}
            eval_config=$(echo "$train_config" | sed "s|_openvlaoft_${tag}$|_openvlaoft_eval_${tag}|")
        fi
    fi
    echo "$eval_config"
}

# Get the default global_step checkpoint index for a given config when MAX_EPOCH is not passed.
# Embodied training saves a final checkpoint at global_step == runner.max_epochs (see embodied_runner).
# Defaults should match typical yaml max_epochs: libero_spatial_grpo_openvlaoft_spatial uses 50.
# For "long" configs (e.g., libero_10_grpo_openvlaoft_long), keep 5 unless you override MAX_EPOCH.
get_default_global_step() {
    local config_name="$1"
    local config_tag
    config_tag=$(extract_config_tag "$config_name")
    if [ "$config_tag" = "long" ]; then
        echo "5"
    else
        echo "50"
    fi
}
