import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForVision2Seq


@torch.no_grad()
def load_base_model_state(base_dir: Path) -> Dict[str, torch.Tensor]:
    """Load HF-exported OpenVLA-OFT model weights from a directory."""
    model = AutoModelForVision2Seq.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    state = model.state_dict()
    # Move everything to CPU / float32 for analysis
    state = {k: v.detach().to("cpu", dtype=torch.float32) for k, v in state.items()}
    return state


def _normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrappers like 'module.' or 'model.' from keys."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]
        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]
        if new_k.startswith("base_model.model."):
            new_k = new_k[len("base_model.model.") :]
        out[new_k] = v
    return out


@torch.no_grad()
def load_finetuned_state(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    """Load a finetuned checkpoint that may be a raw state_dict or a container dict."""
    raw = torch.load(str(ckpt_path), map_location="cpu")

    # Try common wrappers
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
        else:
            # Assume this dict itself is a state dict
            sd = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    sd = _normalize_state_dict_keys(sd)
    sd = {k: v.detach().to("cpu", dtype=torch.float32) for k, v in sd.items()}
    return sd


def is_lora_checkpoint(path: Path) -> bool:
    """Check if path points to a LoRA adapter checkpoint directory."""
    if not path.is_dir():
        return False
    adapter_config = path / "adapter_config.json"
    adapter_model = path / "adapter_model.bin"
    return adapter_config.exists() and adapter_model.exists()


@torch.no_grad()
def load_lora_adapter(
    adapter_dir: Path, base_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Load LoRA adapter weights and reconstruct ΔW = B @ A^T * scaling in original weight space.
    
    Returns a dict mapping original weight names to their ΔW matrices.
    """
    # Load adapter config
    adapter_config_path = adapter_dir / "adapter_config.json"
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    
    lora_rank = adapter_config.get("r", 32)
    lora_alpha = adapter_config.get("lora_alpha", 32)
    scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
    
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}, scaling: {scaling:.4f}")
    
    # Load adapter weights
    adapter_model_path = adapter_dir / "adapter_model.bin"
    if not adapter_model_path.exists():
        # Try safetensors
        adapter_model_path = adapter_dir / "adapter_model.safetensors"
        if adapter_model_path.exists():
            try:
                from safetensors.torch import load_file
                adapter_state = load_file(str(adapter_model_path))
            except ImportError:
                raise ImportError("safetensors not installed. Install with: pip install safetensors")
        else:
            raise FileNotFoundError(f"No adapter_model.bin or adapter_model.safetensors found in {adapter_dir}")
    else:
        adapter_state = torch.load(str(adapter_model_path), map_location="cpu")
    
    # Normalize keys (remove common prefixes)
    adapter_state = _normalize_state_dict_keys(adapter_state)
    
    # Convert to float32
    adapter_state = {k: v.detach().to("cpu", dtype=torch.float32) for k, v in adapter_state.items()}
    
    print(f"  Loaded {len(adapter_state)} adapter weight tensors")
    
    # Group lora_A and lora_B by base module name
    lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    for key, value in adapter_state.items():
        # LoRA keys can have various formats:
        # - "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        # - "language_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        # - "layers.0.self_attn.q_proj.lora_A.default.weight"
        # - "layers.0.self_attn.q_proj.lora_A.weight"
        
        if ".lora_A" in key:
            # Extract base module name (remove .lora_A and everything after)
            if ".lora_A.default.weight" in key:
                base_key = key.split(".lora_A.default.weight")[0]
            elif ".lora_A.weight" in key:
                base_key = key.split(".lora_A.weight")[0]
            elif ".lora_A" in key:
                base_key = key.split(".lora_A")[0]
            else:
                continue
            
            if base_key not in lora_pairs:
                lora_pairs[base_key] = (None, None)
            lora_pairs[base_key] = (value, lora_pairs[base_key][1])
            
        elif ".lora_B" in key:
            if ".lora_B.default.weight" in key:
                base_key = key.split(".lora_B.default.weight")[0]
            elif ".lora_B.weight" in key:
                base_key = key.split(".lora_B.weight")[0]
            elif ".lora_B" in key:
                base_key = key.split(".lora_B")[0]
            else:
                continue
            
            if base_key not in lora_pairs:
                lora_pairs[base_key] = (None, None)
            lora_pairs[base_key] = (lora_pairs[base_key][0], value)
    
    print(f"  Found {len(lora_pairs)} LoRA module pairs")
    
    # Reconstruct ΔW = B @ A^T * scaling for each module
    delta_state: Dict[str, torch.Tensor] = {}
    matched_count = 0
    unmatched_count = 0
    
    for base_key, (lora_A, lora_B) in lora_pairs.items():
        if lora_A is None or lora_B is None:
            unmatched_count += 1
            continue
        
        # LoRA convention: ΔW = B @ A^T * scaling
        # lora_A shape: (rank, in_features) or (in_features, rank) depending on implementation
        # lora_B shape: (out_features, rank) or (rank, out_features)
        # Standard PEFT: lora_A is (rank, in_features), lora_B is (out_features, rank)
        # So ΔW = B @ A^T gives (out_features, in_features)
        
        if lora_A.ndim == 2 and lora_B.ndim == 2:
            # LoRA convention: ΔW = B @ A^T * scaling
            # Standard PEFT: lora_A is (rank, in_features), lora_B is (out_features, rank)
            # So ΔW = B @ A^T gives (out_features, in_features)
            
            # Try standard convention first: B @ A^T
            if lora_B.shape[1] == lora_rank and lora_A.shape[0] == lora_rank:
                # B is (out, rank), A is (rank, in) -> B @ A gives (out, in)
                delta_W = torch.matmul(lora_B, lora_A) * scaling
            elif lora_B.shape[0] == lora_rank and lora_A.shape[1] == lora_rank:
                # B is (rank, out), A is (in, rank) -> B^T @ A^T gives (out, in)
                delta_W = torch.matmul(lora_B.T, lora_A.T) * scaling
            else:
                # Fallback: try B @ A (assuming A might already be transposed in storage)
                delta_W = torch.matmul(lora_B, lora_A) * scaling
        else:
            unmatched_count += 1
            continue
        
        # Map back to original weight name
        # Try various key patterns to match base state
        possible_base_keys = [
            base_key,
            base_key.replace("language_model.model.", ""),
            base_key.replace("language_model.", ""),
            base_key.replace("base_model.model.", ""),
            base_key.replace("base_model.", ""),
        ]
        
        found = False
        for possible_key in possible_base_keys:
            if possible_key in base_state:
                base_weight = base_state[possible_key]
                # Verify shape matches
                if delta_W.shape == base_weight.shape:
                    delta_state[possible_key] = delta_W
                    matched_count += 1
                    found = True
                    break
                elif delta_W.T.shape == base_weight.shape:
                    # Shape was transposed, fix it
                    delta_state[possible_key] = delta_W.T
                    matched_count += 1
                    found = True
                    break
        
        if not found:
            # Store with original key name even if not in base_state
            # (might be a new module added by LoRA)
            delta_state[base_key] = delta_W
            unmatched_count += 1
    
    print(f"  Matched {matched_count} modules to base weights, {unmatched_count} unmatched")
    
    return delta_state


def compute_delta_state(
    base: Dict[str, torch.Tensor], finetuned: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Compute ΔW = W_finetuned − W_base for matching parameters."""
    delta: Dict[str, torch.Tensor] = {}
    for name, wb in base.items():
        wf = finetuned.get(name, None)
        if wf is None:
            continue
        if wb.shape != wf.shape:
            continue
        delta[name] = wf - wb
    return delta


def _tensor_to_matrix(t: torch.Tensor) -> torch.Tensor:
    """Reshape an arbitrary tensor into a 2D matrix for SVD."""
    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        return t.reshape(1, -1)
    # Keep first dim as rows, flatten the rest into columns
    return t.reshape(t.shape[0], -1)


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """Compute effective rank = exp( Shannon entropy of normalized singular values )."""
    if singular_values.size == 0:
        return 0.0
    s = np.abs(singular_values)
    s_sum = s.sum()
    if s_sum <= eps:
        return 0.0
    p = s / s_sum
    # Avoid log(0)
    p = np.clip(p, eps, 1.0)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


@torch.no_grad()
def analyze_delta(
    delta: Dict[str, torch.Tensor], max_params: Optional[int] = None
) -> None:
    """
    Compute SVD-based statistics for each parameter delta (per-layer) and then aggregate globally.
    """
    per_layer_stats: List[Dict[str, float]] = []
    all_singular_values: List[float] = []

    print("Per-layer ΔW statistics:")
    print("-" * 100)
    print(f"{'Layer name':<80s} | {'Shape':<20s} | {'Nuc Norm':<12s} | {'Eff Rank':<10s} | {'Top SV':<12s} | {'Decay':<10s}")
    print("-" * 100)

    for i, (name, dw) in enumerate(delta.items()):
        if max_params is not None and i >= max_params:
            break

        mat = _tensor_to_matrix(dw)
        # Guard against extremely large matrices
        try:
            s = torch.linalg.svdvals(mat)
        except RuntimeError as e:
            print(f"[WARN] SVD failed for {name} with shape {tuple(dw.shape)}: {e}")
            continue

        s_np = s.cpu().numpy()
        all_singular_values.extend(s_np.tolist())

        nuc = float(s_np.sum())
        erank = effective_rank(s_np)
        top_sv = float(s_np[0]) if s_np.size > 0 else 0.0
        tail_sv = float(s_np[-1]) if s_np.size > 0 else 0.0
        spectral_decay = top_sv / (tail_sv + 1e-12) if s_np.size > 1 else 0.0

        per_layer_stats.append({
            "name": name,
            "nuc": nuc,
            "erank": erank,
            "top_sv": top_sv,
            "spectral_decay": spectral_decay,
            "shape": tuple(dw.shape),
        })

        print(
            f"{name:<80s} | {str(tuple(dw.shape)):<20s} | "
            f"{nuc:>12.4e} | {erank:>10.2f} | "
            f"{top_sv:>12.4e} | {spectral_decay:>10.2e}"
        )

    print("\n" + "=" * 100)
    print("Aggregated statistics (across all layers):")
    print("=" * 100)
    
    if not all_singular_values:
        print("No singular values computed (empty ΔW).")
        return

    # Global statistics from all singular values
    all_s = np.array(all_singular_values, dtype=np.float64)
    global_nuc = float(all_s.sum())
    global_erank = effective_rank(all_s)
    all_s_sorted = np.sort(all_s)[::-1]
    top1 = float(all_s_sorted[0])
    frac_top10 = float(all_s_sorted[:10].sum() / (all_s_sorted.sum() + 1e-12))

    print(f"Nuclear norm (sum of all singular values):     {global_nuc:.4e}")
    print(f"Effective rank (global):                      {global_erank:.2f}")
    print(f"Largest singular value:                       {top1:.4e}")
    print(f"Fraction of nuclear norm in top-10 svs:        {frac_top10:.4f}")
    
    # Per-layer aggregation statistics
    if per_layer_stats:
        eranks = [s["erank"] for s in per_layer_stats]
        nucs = [s["nuc"] for s in per_layer_stats]
        
        print("\n" + "=" * 100)
        print("Per-layer aggregation statistics:")
        print("=" * 100)
        print(f"Mean effective rank per layer:              {np.mean(eranks):.2f}")
        print(f"Median effective rank per layer:           {np.median(eranks):.2f}")
        print(f"Std effective rank per layer:               {np.std(eranks):.2f}")
        print(f"Min effective rank per layer:              {np.min(eranks):.2f}")
        print(f"Max effective rank per layer:             {np.max(eranks):.2f}")
        print(f"Total nuclear norm (sum across layers):    {np.sum(nucs):.4e}")
        print(f"Mean nuclear norm per layer:                {np.mean(nucs):.4e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ΔW between base OpenVLA-OFT weights and a finetuned checkpoint (full FT or LoRA), "
        "then run SVD-based diagnostics (effective rank, spectral decay, nuclear norm) per-layer and globally."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=False,
        default="/scratch/cluster/jhu/RLinf/model/Openvla-oft-SFT-libero-spatial-traj1",
        help="Path to HF-exported base OpenVLA-OFT directory (with config.json, model-*.safetensors, etc.).",
    )
    parser.add_argument(
        "--finetuned_ckpt",
        type=str,
        required=True,
        help="Path to finetuned checkpoint (.pt file for full FT, or directory with adapter_model.bin for LoRA).",
    )
    parser.add_argument(
        "--max_params",
        type=int,
        default=None,
        help="Optional cap on number of parameters to analyze (for quick tests).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    finetuned_path = Path(args.finetuned_ckpt)

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Finetuned checkpoint not found: {finetuned_path}")

    print(f"Loading base model state from:    {base_dir}")
    base_state = load_base_model_state(base_dir)

    # Check if finetuned_path is a LoRA adapter directory
    if is_lora_checkpoint(finetuned_path):
        print(f"Detected LoRA adapter checkpoint: {finetuned_path}")
        print("Loading LoRA adapter weights and reconstructing ΔW = A @ B^T * scaling...")
        delta_state = load_lora_adapter(finetuned_path, base_state)
    else:
        print(f"Loading finetuned state from:     {finetuned_path}")
        finetuned_state = load_finetuned_state(finetuned_path)
        print("Computing ΔW = W_finetuned − W_base ...")
        delta_state = compute_delta_state(base_state, finetuned_state)

    if not delta_state:
        print("ΔW is empty: no overlapping parameters between base and finetuned checkpoints.")
        return

    print(f"\nFound {len(delta_state)} parameter matrices to analyze.\n")
    analyze_delta(delta_state, max_params=args.max_params)


if __name__ == "__main__":
    main()
