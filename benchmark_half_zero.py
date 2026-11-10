"""Zero the three saved half-split masks; reuse native and Granite measurements."""

import argparse
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, __version__

from benchmark_half_split import CONTEXTS, FOREIGN_K, SEEDS, TOP_K
from benchmark_substitution import CONTEXT, ROOT, START, STOP, write_json
from co_moe0 import PRIMARY, PRIMARY_REV

SOURCE = ROOT / "results/half-split-benchmark"


class Settings(BaseModel):
    """Only the fresh output directory is configurable."""

    output: Path = Path("results/half-split-zero")


class Reference(BaseModel):
    """Read the completed comparison receipt without altering its measurements."""

    outcome: Literal["COMPLETED"]
    protocol: dict
    runs: dict[str, dict]


@torch.inference_mode()
def main():
    """Run exactly three matched-zero conditions on the previously fixed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Settings().output)
    settings = Settings.model_validate(vars(parser.parse_args()))
    settings.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)
    source = Reference.model_validate_json((SOURCE / "receipt.json").read_text())
    masks = torch.load(SOURCE / "masks.pt", weights_only=True)
    ids = torch.load(SOURCE / "input_ids.pt", weights_only=True)
    routing = torch.load(SOURCE / "native_routing.pt", weights_only=True)
    if ids.shape != (1, STOP - START):
        raise ValueError("The exact 4096 saved input IDs are required")
    for seed in SEEDS:
        mask = masks[f"split_seed_{seed}"]
        if mask.shape != (CONTEXTS, CONTEXT, TOP_K) or not bool((mask.sum(-1) == FOREIGN_K).all()):
            raise ValueError("Each saved mask must select four of eight positions per token")
    protocol = {
        "experiment": "Matched-zero controls for the three saved layer-7 half-split masks",
        "fixed_before_scoring_utc": datetime.now(UTC).isoformat(),
        "reference_receipt": "results/half-split-benchmark/receipt.json",
        "reference_conditions_reused": list(source.runs),
        "primary": {"model": PRIMARY, "revision": PRIMARY_REV, "layer": 7},
        "data": source.protocol["data"],
        "saved_masks": "results/half-split-benchmark/masks.pt",
        "mask_seeds": list(SEEDS),
        "mask_generation": False,
        "conditions_executed": [f"zero_seed_{seed}" for seed in SEEDS],
        "replacement": "Set the same four selected raw outputs to zero before original weighting",
        "routing": "Original indices and weights; no renormalization of the remaining slots",
        "donor_loaded_or_executed": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __version__,
            "device": "cpu",
            "dtype": "float32",
            "loss_accumulation": "float64 sum of float32 cross-entropy values",
            "seed": 0,
            "threads": 12,
            "attention": "eager",
            "experts": "eager",
            "use_cache": False,
        },
        "comparison": "Granite minus zero, paired by saved mask and reference-token position",
        "training": False,
        "tuning": False,
        "limitations": [
            "Three mask draws on one previously evaluated literary slice",
            "A benefit over zero would not establish a benefit from pretrained structure",
            "No random-weight control at this density, no coefficient or mask search",
            "No new native/Granite run or numerical restoration run",
        ],
    }
    write_json(settings.output / "protocol.json", protocol)
    torch.save(masks, settings.output / "masks.pt")
    torch.save(ids, settings.output / "input_ids.pt")
    print("Protocol recorded; reusing saved masks, inputs, and native/Granite scores.", flush=True)
    path = snapshot_download(
        PRIMARY, revision=PRIMARY_REV, allow_patterns=["*.json", "*.safetensors"]
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
        )
        .eval()
        .requires_grad_(False)
    )
    experts = model.model.layers[7].mlp.experts
    original = experts.forward
    observed = {}

    def zeroed(module, hidden_states, indices, weights):
        """Apply zeros at saved top-k positions, retaining the native eager sum order."""
        selected = masks[source_name][window]
        indices_equal = torch.equal(indices, routing["indices"][window])
        weight_error = float((weights - routing["weights"][window]).abs().max())
        if not indices_equal or weight_error != 0:
            raise RuntimeError("Layer-7 routing differs from the saved native reference")
        final = torch.zeros_like(hidden_states)
        native_sum = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(indices, num_classes=module.num_experts + 1).permute(2, 1, 0)
        count = 0
        kept_error = zero_error = 0.0
        for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
            idx = expert_idx[0]
            if idx == module.num_experts:
                continue
            slots, positions = torch.where(expert_mask[idx])
            x = hidden_states[positions]
            gate, up = F.linear(x, module.gate_up_proj[idx]).chunk(2, -1)
            native_raw = F.linear(module.act_fn(gate) * up, module.down_proj[idx])
            replace = selected[positions, slots]
            raw = torch.where(replace[:, None], torch.zeros_like(native_raw), native_raw)
            if not bool(torch.isfinite(raw).all() and torch.isfinite(native_raw).all()):
                raise FloatingPointError("Nonfinite raw expert output")
            if bool((~replace).any()):
                kept_error = max(
                    kept_error, float((raw[~replace] - native_raw[~replace]).abs().max())
                )
            if bool(replace.any()):
                zero_error = max(zero_error, float(raw[replace].abs().max()))
            count += int(replace.sum())
            mixture = weights[positions, slots, None]
            final.index_add_(0, positions, (raw * mixture).to(final.dtype))
            native_sum.index_add_(0, positions, (native_raw * mixture).to(final.dtype))
        if count != CONTEXT * FOREIGN_K or kept_error != 0 or zero_error != 0:
            raise RuntimeError("The saved half mask was not applied exactly")
        observed.update(
            routing_indices_equal=indices_equal,
            routing_weights_max_abs=weight_error,
            kept_raw_max_abs_vs_computed_native=kept_error,
            zero_raw_max_abs=zero_error,
            zeroed_contributions=count,
            native_contributions=CONTEXT * TOP_K - count,
            masked_routing_weight_sum=float(weights[selected].double().sum()),
            total_routing_weight_sum=float(weights.double().sum()),
            moe_delta_l2_vs_local_native_sum=float((final - native_sum).norm()),
            finite=bool(torch.isfinite(final).all()),
        )
        return final

    runs, losses = {}, {}
    receipt = {"protocol": protocol, "outcome": "PARTIAL", "runs": runs}
    write_json(settings.output / "receipt.json", receipt)
    experts.forward = MethodType(zeroed, experts)
    try:
        for seed in SEEDS:
            name, source_name = f"zero_seed_{seed}", f"split_seed_{seed}"
            windows, token_losses = [], []
            started = time.monotonic()
            for window, offset in enumerate(range(0, STOP - START, CONTEXT)):
                observed = {}
                output = model(input_ids=ids[:, offset : offset + CONTEXT], use_cache=False)
                loss = F.cross_entropy(
                    output.logits[0, :-1], ids[0, offset + 1 : offset + CONTEXT], reduction="none"
                ).double()
                if (
                    not bool(torch.isfinite(output.logits).all() and torch.isfinite(loss).all())
                    or not observed["finite"]
                ):
                    raise FloatingPointError(f"Nonfinite {name}, context {window}")
                token_losses.append(loss.clone())
                nll = float(loss.mean())
                windows.append(
                    {
                        "context": window,
                        "corpus_range": [START + offset, START + offset + CONTEXT],
                        "scored_targets": loss.numel(),
                        "nll_sum": float(loss.sum()),
                        "mean_nll": nll,
                        "delta_nll_vs_native": nll
                        - source.runs["native"]["windows"][window]["mean_nll"],
                        "granite_minus_zero_nll": source.runs[source_name]["windows"][window][
                            "mean_nll"
                        ]
                        - nll,
                        **observed,
                    }
                )
                if (window + 1) % 4 == 0:
                    print(f"{name}: {window + 1}/{CONTEXTS} contexts completed", flush=True)
            losses[name] = torch.stack(token_losses)
            nll = float(losses[name].mean())
            delta = nll - source.runs["native"]["mean_nll"]
            paired = source.runs[source_name]["mean_nll"] - nll
            runs[name] = {
                "mask_seed": seed,
                "reference_granite_condition": source_name,
                "mean_nll": nll,
                "perplexity": math.exp(nll),
                "delta_nll_vs_native": delta,
                "perplexity_change_percent_vs_native": 100 * math.expm1(delta),
                "granite_minus_zero_nll": paired,
                "granite_perplexity_change_percent_vs_zero": 100 * math.expm1(paired),
                "scored_targets": losses[name].numel(),
                "zeroed_contributions": sum(w["zeroed_contributions"] for w in windows),
                "native_contributions": sum(w["native_contributions"] for w in windows),
                "masked_routing_weight_fraction": sum(
                    w["masked_routing_weight_sum"] for w in windows
                )
                / sum(w["total_routing_weight_sum"] for w in windows),
                "finite": all(w["finite"] for w in windows),
                "seconds": round(time.monotonic() - started, 2),
                "windows": windows,
            }
            torch.save(losses, settings.output / "loss_per_token.pt")
            write_json(settings.output / "receipt.json", receipt)
            print(
                f"{name}: NLL={nll:.9f}, PPL={math.exp(nll):.9f}; Granite-zero NLL={paired:+.9f}",
                flush=True,
            )
    finally:
        experts.forward = original
    receipt["native_forward_method_restored"] = experts.forward == original
    receipt["outcome"] = "COMPLETED"
    write_json(settings.output / "receipt.json", receipt)
    print(f"COMPLETED: {settings.output / 'receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
