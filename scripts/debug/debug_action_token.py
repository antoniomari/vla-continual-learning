import numpy as np

# ============================================================
# Hardcoded configuration (must exactly match training)
# ============================================================

CHUNK_SIZE = 8
ACTION_DIM = 7
VOCAB_SIZE = 32000
N_ACTION_BINS = 256

# --- Bin centers (EXACT match) ---
BINS = np.linspace(-1.0, 1.0, N_ACTION_BINS)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0  # (255,)

# NOTE:
# Training discretizes into (n_action_bins - 1) centers
# Tokens are clipped into [vocab - n_action_bins, vocab - 1]

# --- Normalization type ---
NORMALIZATION_TYPE = "BOUNDS_Q99"

# --- Action normalization stats ---
ACTION_STATS = {
    "max": np.array(
        [
            0.9375,
            0.9053571224212646,
            0.9375,
            0.14249999821186066,
            0.20571428537368774,
            0.08464285731315613,
            1.0,
        ]
    ),
    "min": np.array(
        [
            -0.7901785969734192,
            -0.8303571343421936,
            -0.9375,
            -0.15214285254478455,
            -0.24535714089870453,
            -0.21857142448425293,
            0.0,
        ]
    ),
    "q99": np.array(
        [
            0.9375,
            0.8758928775787354,
            0.8472589308023452,
            0.11423571482300747,
            0.15461785525083535,
            0.06997500024735921,
            1.0,
        ]
    ),
    "q01": np.array(
        [
            -0.7168392646312713,
            -0.6970714360475541,
            -0.9375,
            -0.12107142806053162,
            -0.22056428104639053,
            -0.18339642822742463,
            0.0,
        ]
    ),
    "mask": np.array([True, True, True, True, True, True, False]),
}


# ============================================================
# Normalization (EXACT mirror of training code)
# ============================================================
def unnormalize_actions(normalized_actions):
    """Unnormalize actions using dataset statistics"""
    if NORMALIZATION_TYPE == "BOUNDS":
        mask = ACTION_STATS.get("mask", np.ones_like(ACTION_STATS["min"], dtype=bool))
        action_high, action_low = (
            np.array(ACTION_STATS["max"]),
            np.array(ACTION_STATS["min"]),
        )
    elif NORMALIZATION_TYPE == "BOUNDS_Q99":
        mask = ACTION_STATS.get("mask", np.ones_like(ACTION_STATS["q01"], dtype=bool))
        action_high, action_low = (
            np.array(ACTION_STATS["q99"]),
            np.array(ACTION_STATS["q01"]),
        )
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return actions


def normalize_actions(actions):
    """
    actions: (B, T, D) or (N, D)
    returns normalized actions in [-1, 1]
    """
    actions = np.asarray(actions)

    if NORMALIZATION_TYPE == "BOUNDS":
        mask = ACTION_STATS.get("mask", np.ones_like(ACTION_STATS["min"], dtype=bool))
        action_high, action_low = (
            np.array(ACTION_STATS["max"]),
            np.array(ACTION_STATS["min"]),
        )
    elif NORMALIZATION_TYPE == "BOUNDS_Q99":
        mask = ACTION_STATS.get("mask", np.ones_like(ACTION_STATS["q01"], dtype=bool))
        action_high, action_low = (
            np.array(ACTION_STATS["q99"]),
            np.array(ACTION_STATS["q01"]),
        )
    else:
        raise ValueError("Unsupported normalization type")

    action_dim = actions.shape[-1]
    repeat_factor = action_dim // action_high.shape[0]

    action_high = action_high.repeat(repeat_factor)
    action_low = action_low.repeat(repeat_factor)
    mask = mask * repeat_factor

    normalized_actions = np.where(
        mask,
        2.0 * (actions - action_low) / (action_high - action_low) - 1.0,
        actions,
    )

    return normalized_actions


# ============================================================
# Actions → action tokens (EXACT inverse of decode)
# ============================================================
def compute_actions_from_action_tokens(action_tokens):
    action_tokens = action_tokens.reshape(-1, ACTION_DIM)
    predicted_action_token_ids = action_tokens
    discretized_actions = VOCAB_SIZE - predicted_action_token_ids
    print("disc afction", discretized_actions)
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=BIN_CENTERS.shape[0] - 1
    )
    print("clipped disc action", discretized_actions)
    # normalized_actions = model.bin_centers[discretized_actions]
    normalized_actions = np.asarray(
        [BIN_CENTERS[da] for da in discretized_actions]
    )  # [B, dim]
    normalized_actions = normalized_actions.reshape(-1, ACTION_DIM)
    print("noramlied actions", normalized_actions)

    # Unnormalize predicted actions
    actions = unnormalize_actions(normalized_actions)
    actions = normalized_actions
    actions = actions.reshape(-1, CHUNK_SIZE, ACTION_DIM)
    return actions


# def compute_action_tokens_from_actions(actions):
#     """
#     actions: (B, T, D)
#     returns: (B, T*D) absolute action tokens
#     """
#     actions = np.asarray(actions)
#     B, T, D = actions.shape
#     assert D == ACTION_DIM

#     # Normalize
#     normalized_actions = normalize_actions(actions)
#     normalized_actions = normalized_actions.reshape(-1, D)  # (B*T, D)

#     discretized_actions = []
#     for dim in range(D):
#         vals = normalized_actions[:, dim]  # (B*T,)
#         nearest_bins = np.searchsorted(BINS, vals, side="right") - 1
#         discretized_actions.append(nearest_bins)

#     discretized_actions = np.stack(discretized_actions, axis=1)  # (B*T, D)

#     token_ids = VOCAB_SIZE - 1 - discretized_actions
#     token_ids = token_ids.reshape(B, T * D)
#     return token_ids


def compute_action_tokens_from_actions(actions):
    """
    Inverse of the action tokens to continuous actions


    chunk_action_tokens = idxs.reshape(-1, model.action_dim)
    predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1
    )
    # normalized_actions = model.bin_centers[discretized_actions]
    normalized_actions = np.asarray(
        [model.bin_centers[da] for da in discretized_actions]
    )  # [B, dim]
    normalized_actions = normalized_actions.reshape(-1, model.action_dim)

    # Unnormalize predicted actions
    actions = model._unnormalize_actions(normalized_actions, model.unnorm_key)
    actions = actions.reshape(idxs.shape)
    """

    B, T, D = actions.shape

    normalized_actions = normalize_actions(actions)
    normalized_actions = actions
    normalized_actions = normalized_actions.reshape(-1, D)
    bin_centers = BIN_CENTERS

    discretized_actions = []
    for dim in range(D):
        vals = normalized_actions[:, dim][:, None]  # (B*T, 1)
        print(dim, vals)
        dists = np.abs(vals - bin_centers[None, :])  # (B*T, n_bins)
        nearest_bins = np.argmin(dists, axis=1)  # (B*T,)
        discretized_actions.append(nearest_bins)

    discretized_actions = np.stack(discretized_actions, axis=1)  # (B*T, D)

    token_ids = VOCAB_SIZE - 1 - discretized_actions
    token_ids = np.clip(
        token_ids,
        VOCAB_SIZE - N_ACTION_BINS,
        VOCAB_SIZE - 1,
    )

    token_ids = token_ids.reshape(B, T * D)
    return token_ids


# ============================================================
# Verification script
# ============================================================


def verify_action_token_roundtrip(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    dataset_actions = data["actions"]  # (B, T, D) continuous
    dataset_tokens = data["action_tokens"]  # (B, T*D) or (B, T, D)

    if dataset_tokens.ndim == 3:
        B, T, D = dataset_tokens.shape
        dataset_tokens = dataset_tokens.reshape(B, T * D)

    B, TD = dataset_tokens.shape
    T = TD // ACTION_DIM

    # Recompute tokens from dataset actions
    recomputed_tokens = compute_action_tokens_from_actions(dataset_actions)

    # Accuracy
    token_match = dataset_tokens == recomputed_tokens
    overall_acc = token_match.mean()

    # Reshape for analysis
    dataset_tokens_reshaped = dataset_tokens.reshape(B, T, ACTION_DIM)
    recomputed_tokens_reshaped = recomputed_tokens.reshape(B, T, ACTION_DIM)

    # Print per-dimension accuracy first
    print("\n=== Per-dimension token accuracy ===")
    for d in range(ACTION_DIM):
        acc = (
            dataset_tokens_reshaped[:, :, d] == recomputed_tokens_reshaped[:, :, d]
        ).mean()
        print(f"Dim {d}: {acc * 100:.6f}%")

    # Print overall accuracy
    print(f"\nOverall token match accuracy: {overall_acc * 100:.6f}%")

    # Print mismatch details
    print("\n================ MISMATCH DETAILS ================")

    for d in range(ACTION_DIM):
        mismatches = np.argwhere(
            dataset_tokens_reshaped[:, :, d] != recomputed_tokens_reshaped[:, :, d]
        )

        if mismatches.size == 0:
            print(f"\nDim {d}: ✅ no mismatches")
            continue

        # Compute decoded action differences for all mismatches
        mismatch_diffs = []
        for idx in range(mismatches.shape[0]):
            b, t = mismatches[idx]

            true_action = dataset_actions[b, t, d]
            dataset_token = dataset_tokens_reshaped[b, t, d]
            recomputed_token = recomputed_tokens_reshaped[b, t, d]

            # Decode both tokens
            dataset_bin = dataset_token - (VOCAB_SIZE - N_ACTION_BINS)
            recomputed_bin = recomputed_token - (VOCAB_SIZE - N_ACTION_BINS)

            dataset_normalized = BIN_CENTERS[dataset_bin]
            recomputed_normalized = BIN_CENTERS[recomputed_bin]

            temp_vec_dataset = np.zeros(ACTION_DIM)
            temp_vec_dataset[d] = dataset_normalized
            decoded_from_dataset = unnormalize_actions(temp_vec_dataset)[d]

            temp_vec_recomputed = np.zeros(ACTION_DIM)
            temp_vec_recomputed[d] = recomputed_normalized
            decoded_from_recomputed = unnormalize_actions(temp_vec_recomputed)[d]

            # Calculate actual decoded difference
            decoded_diff = abs(decoded_from_dataset - decoded_from_recomputed)

            mismatch_diffs.append(
                {
                    "b": b,
                    "t": t,
                    "true_action": true_action,
                    "dataset_token": dataset_token,
                    "recomputed_token": recomputed_token,
                    "decoded_dataset": decoded_from_dataset,
                    "decoded_recomputed": decoded_from_recomputed,
                    "decoded_diff": decoded_diff,
                    "token_diff": abs(int(dataset_token) - int(recomputed_token)),
                }
            )

        # Sort by decoded difference (largest first)
        mismatch_diffs.sort(key=lambda x: x["decoded_diff"], reverse=True)

        print(
            f"\nDim {d}: ❌ {len(mismatch_diffs)} mismatches (showing top 3 by decoded difference)"
        )

        for i in range(min(3, len(mismatch_diffs))):
            m = mismatch_diffs[i]
            print(f"\n  Mismatch #{i + 1}: Batch {m['b']}, Timestep {m['t']}, Dim {d}")
            print(f"    True action (from dataset)    : {m['true_action']:.9f}")
            print(f"    Dataset token                 : {m['dataset_token']}")
            print(f"    Recomputed token              : {m['recomputed_token']}")
            print(f"    Token difference              : {m['token_diff']}")
            print(f"    Decoded (dataset token)       : {m['decoded_dataset']:.9f}")
            print(f"    Decoded (recomputed token)    : {m['decoded_recomputed']:.9f}")
            print(f"    Decoded action difference     : {m['decoded_diff']:.9f}")

    return overall_acc


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    # verify_action_token_roundtrip("./scripts/debug/bcrl_wrong.npz")
    # verify_action_token_roundtrip("./scripts/debug/base.npz")

    action_tokens = np.full((1, CHUNK_SIZE, ACTION_DIM), 31744, dtype=np.int64)

    print("\n================ SANITY CHECK =================")
    print("Action tokens:")
    print(action_tokens)
    print("Shape:", action_tokens.shape)

    # --------------------------------------------------------
    # 2. Tokens -> actions (includes unnormalization internally)
    # --------------------------------------------------------
    actions = compute_actions_from_action_tokens(action_tokens)

    print("\nActions after token → action (UNNORMALIZED):")
    print(actions)
    print("Min / Max:", actions.min(), actions.max())

    # --------------------------------------------------------
    # 3. Normalize those actions
    # --------------------------------------------------------
    normalized = normalize_actions(actions)

    print("\nNormalized actions:")
    print(normalized)
    print("Min / Max:", normalized.min(), normalized.max())

    # --------------------------------------------------------
    # 4. Recompute tokens from normalized→unnormalized actions
    # --------------------------------------------------------
    recomputed_tokens = compute_action_tokens_from_actions(actions)

    print("\nRecomputed action tokens (FULL roundtrip):")
    print(recomputed_tokens.reshape(1, CHUNK_SIZE, ACTION_DIM))

    # --------------------------------------------------------
    # 5. Direct roundtrip (skip normalize / unnormalize mentally)
    #    token -> action -> token
    # --------------------------------------------------------
    forward = compute_actions_from_action_tokens(action_tokens)
    direct_roundtrip_tokens = compute_action_tokens_from_actions(forward)
    # print(forward, direct_roundtrip_tokens)

    print("\nDirect token → action → token:")
    print(direct_roundtrip_tokens.reshape(1, CHUNK_SIZE, ACTION_DIM))

    print("\nToken differences:")
    print(direct_roundtrip_tokens.reshape(1, CHUNK_SIZE, ACTION_DIM) - action_tokens)

    print("\n================ END SANITY CHECK =================")
