"""Three fixed 50/50 native/Granite contribution masks at OLMoE layer 7."""

import argparse
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, __version__

from benchmark_substitution import CONTEXT, ROOT, START, STOP, write_json
from co_moe0 import PRIMARY, PRIMARY_REV
from fit_native_expert import CORPUS_REV, CORPUS_URL
from reciprocal_inference import GRANITE, GRANITE_REV, project
from replicate_granite_pair import execute, granite_expert

SEEDS = (0, 1, 2)
TOP_K, FOREIGN_K = 8, 4
CONTEXTS = (STOP - START) // CONTEXT


class Settings(BaseModel):
    """Only the output destination varies; data and three mask seeds stay fixed."""

    output: Path = Path("results/half-split-benchmark")


@torch.inference_mode()
def main():
    """Run a native reference and three randomly positioned half substitutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Settings().output)
    settings = Settings.model_validate(vars(parser.parse_args()))
    settings.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)
    protocol = {
        "experiment": "50/50 routed contribution substitution at OLMoE layer 7",
        "fixed_before_scoring_utc": datetime.now(UTC).isoformat(),
        "primary": {"model": PRIMARY, "revision": PRIMARY_REV, "layer": 7},
        "donor": {"model": GRANITE, "revision": GRANITE_REV, "layer": 7, "expert": 0},
        "data": {
            "corpus": "Tiny Shakespeare",
            "revision": CORPUS_REV,
            "url": CORPUS_URL,
            "source_token_ids": "results/substitution-benchmark/input_ids.pt",
            "token_range": [START, STOP],
            "selection": "Reuse the preceding benchmark slice; no new text selection",
            "context_tokens": CONTEXT,
            "contexts": CONTEXTS,
            "input_tokens": STOP - START,
            "scored_targets": CONTEXTS * (CONTEXT - 1),
            "scoring": "Teacher forcing: logits[j] score reference[j+1], j=0..254; reset each context",
        },
        "conditions": ["native", *[f"split_seed_{seed}" for seed in SEEDS]],
        "mask": {
            "seeds": list(SEEDS),
            "method": "First 4 positions of an independent randperm(8) per token",
            "shape": [CONTEXTS, CONTEXT, TOP_K],
            "selection_unit": "Native top-k position, independently per context/token",
            "foreign_per_token": FOREIGN_K,
            "native_per_token": TOP_K - FOREIGN_K,
            "foreign_contributions_per_run": (STOP - START) * FOREIGN_K,
            "native_contributions_per_run": (STOP - START) * (TOP_K - FOREIGN_K),
            "fixed_before_model_loading": True,
        },
        "replacement": "Replace only masked raw expert outputs before original mixture weighting",
        "projections": "Truncate input 2048 -> 1024; zero-pad output 1024 -> 2048",
        "same_donor_for_all_selected_slots": True,
        "donor_evaluation": "Once per token, reused at its four masked positions",
        "extra_alpha": None,
        "norm_matching": False,
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
        "training": False,
        "tuning": False,
        "limitations": [
            "50 percent of routed positions in one layer, not of the whole model",
            "Equal counts do not imply equal routing weight, output norm, or computation cost",
            "One pretrained donor expert reused, not four distinct donor experts per token",
            "Three mask draws on the same literary slice are not independent datasets",
            "No matched zero/random-weight controls at this replacement density",
            "No general quality, pretraining-transfer, or speed claim",
        ],
    }
    write_json(settings.output / "protocol.json", protocol)
    masks = {}
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.stack(
            [torch.randperm(TOP_K, generator=generator)[:FOREIGN_K] for _ in range(STOP - START)]
        ).reshape(CONTEXTS, CONTEXT, FOREIGN_K)
        mask = torch.zeros((CONTEXTS, CONTEXT, TOP_K), dtype=torch.bool)
        mask.scatter_(-1, selected, True)
        assert bool((mask.sum(-1) == FOREIGN_K).all())
        masks[f"split_seed_{seed}"] = mask
    torch.save(masks, settings.output / "masks.pt")
    ids = torch.load(ROOT / protocol["data"]["source_token_ids"], weights_only=True)
    if ids.shape != (1, STOP - START):
        raise ValueError("The previous benchmark's exact 4096 input token IDs are required")
    torch.save(ids, settings.output / "input_ids.pt")
    print("Protocol and all three 50/50 masks fixed before model loading.", flush=True)

    donor, _ = granite_expert()
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
    baseline, scores, runs = [], {}, {}
    observed = {}

    def routed(module, hidden_states, indices, weights):
        """Retain the native eager sum order and replace four raw outputs per token."""
        selected = (
            torch.zeros_like(indices, dtype=torch.bool)
            if condition == "native"
            else masks[condition][window]
        )
        foreign = None
        if condition != "native":
            foreign = project(execute(donor, hidden_states[..., :1024]), 2048)
            if not bool(torch.isfinite(foreign).all()):
                raise FloatingPointError("Nonfinite Granite response")
        final = torch.zeros_like(hidden_states)
        applied = hidden_states.new_empty((len(hidden_states), TOP_K, hidden_states.shape[-1]))
        expert_mask = F.one_hot(indices, num_classes=module.num_experts + 1).permute(2, 1, 0)
        for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
            idx = expert_idx[0]
            if idx == module.num_experts:
                continue
            slots, positions = torch.where(expert_mask[idx])
            x = hidden_states[positions]
            gate, up = F.linear(x, module.gate_up_proj[idx]).chunk(2, -1)
            raw = F.linear(module.act_fn(gate) * up, module.down_proj[idx])
            if foreign is not None:
                raw = torch.where(selected[positions, slots, None], foreign[positions], raw)
            applied[positions, slots] = raw
            final.index_add_(0, positions, (raw * weights[positions, slots, None]).to(final.dtype))

        native_error = kept_error = foreign_error = 0.0
        if condition == "native":
            native_output = original(hidden_states, indices, weights)
            native_error = float((final - native_output).abs().max())
            if native_error != 0:
                raise RuntimeError("The recorded native eager sum differs from the original")
            final = native_output
        else:
            kept_error = float(
                (applied[~selected] - baseline[window]["raw"][~selected]).abs().max()
            )
            expected = foreign[:, None, :].expand_as(applied)
            foreign_error = float((applied[selected] - expected[selected]).abs().max())
            if kept_error != 0 or foreign_error != 0:
                raise RuntimeError("A kept native or replaced foreign raw output differs")
        observed.update(
            inputs=hidden_states.clone(),
            indices=indices.clone(),
            weights=weights.clone(),
            raw=applied,
            moe_output=final.clone(),
            selected=selected,
            native_reconstruction_max_abs=native_error,
            kept_native_raw_max_abs=kept_error,
            foreign_raw_max_abs=foreign_error,
        )
        return final

    receipt = {"protocol": protocol, "outcome": "PARTIAL", "runs": runs}
    write_json(settings.output / "receipt.json", receipt)
    experts.forward = MethodType(routed, experts)
    try:
        for condition in protocol["conditions"]:
            windows, losses = [], []
            started = time.monotonic()
            for window, offset in enumerate(range(0, STOP - START, CONTEXT)):
                observed = {}
                output = model(input_ids=ids[:, offset : offset + CONTEXT], use_cache=False)
                loss = F.cross_entropy(
                    output.logits[0, :-1], ids[0, offset + 1 : offset + CONTEXT], reduction="none"
                ).double()
                if not all(
                    bool(torch.isfinite(t).all())
                    for t in [output.logits, loss, observed["raw"], observed["moe_output"]]
                ):
                    raise FloatingPointError(f"Nonfinite {condition}, context {window}")
                losses.append(loss.clone())
                if condition == "native":
                    baseline.append(observed)
                native = baseline[window]
                same_indices = torch.equal(observed["indices"], native["indices"])
                weight_error = float((observed["weights"] - native["weights"]).abs().max())
                input_error = float((observed["inputs"] - native["inputs"]).abs().max())
                if not same_indices or weight_error != 0 or input_error != 0:
                    raise RuntimeError("Layer-7 routing or its pre-intervention inputs changed")
                selected = observed["selected"]
                nll = float(loss.mean())
                windows.append(
                    {
                        "context": window,
                        "corpus_range": [START + offset, START + offset + CONTEXT],
                        "scored_targets": loss.numel(),
                        "nll_sum": float(loss.sum()),
                        "mean_nll": nll,
                        "delta_nll_vs_native": nll
                        - (
                            runs["native"]["windows"][window]["mean_nll"]
                            if condition != "native"
                            else nll
                        ),
                        "foreign_contributions": int(selected.sum()),
                        "native_contributions": int((~selected).sum()),
                        "foreign_contributions_with_scored_target": int(selected[:-1].sum()),
                        "tokens_with_foreign": int(selected.any(-1).sum()),
                        "foreign_routing_weight_sum": float(
                            observed["weights"][selected].double().sum()
                        ),
                        "total_routing_weight_sum": float(observed["weights"].double().sum()),
                        "routing_indices_equal": same_indices,
                        "routing_weights_max_abs": weight_error,
                        "expert_input_max_abs": input_error,
                        "kept_native_raw_max_abs": observed["kept_native_raw_max_abs"],
                        "foreign_raw_max_abs": observed["foreign_raw_max_abs"],
                        "native_reconstruction_max_abs": observed["native_reconstruction_max_abs"],
                        "moe_delta_l2": float(
                            (observed["moe_output"] - native["moe_output"]).norm()
                        ),
                        "donor_batch_calls": int(condition != "native"),
                        "finite": True,
                    }
                )
                if (window + 1) % 4 == 0:
                    print(f"{condition}: {window + 1}/{CONTEXTS} contexts completed", flush=True)
            scores[condition] = torch.stack(losses)
            mean_nll = float(scores[condition].mean())
            delta = mean_nll - (runs["native"]["mean_nll"] if condition != "native" else mean_nll)
            runs[condition] = {
                "mask_seed": None if condition == "native" else int(condition.rsplit("_", 1)[1]),
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll),
                "delta_nll_vs_native": delta,
                "perplexity_change_percent_vs_native": 100 * math.expm1(delta),
                "scored_targets": scores[condition].numel(),
                "foreign_contributions": sum(w["foreign_contributions"] for w in windows),
                "native_contributions": sum(w["native_contributions"] for w in windows),
                "tokens_with_foreign": sum(w["tokens_with_foreign"] for w in windows),
                "foreign_routing_weight_fraction": sum(
                    w["foreign_routing_weight_sum"] for w in windows
                )
                / sum(w["total_routing_weight_sum"] for w in windows),
                "finite": True,
                "seconds": round(time.monotonic() - started, 2),
                "windows": windows,
            }
            torch.save(scores, settings.output / "loss_per_token.pt")
            write_json(settings.output / "receipt.json", receipt)
            print(f"{condition}: NLL={mean_nll:.9f}, PPL={math.exp(mean_nll):.9f}", flush=True)
    finally:
        experts.forward = original

    torch.save(
        {
            "indices": torch.stack([b["indices"] for b in baseline]),
            "weights": torch.stack([b["weights"] for b in baseline]),
        },
        settings.output / "native_routing.pt",
    )
    receipt["native_forward_method_restored"] = experts.forward == original
    receipt["outcome"] = "COMPLETED"
    write_json(settings.output / "receipt.json", receipt)
    print(f"COMPLETED: {settings.output / 'receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
