"""One fixed teacher-forced OLMoE/Granite substitution comparison, with no tuning."""

import argparse
import copy
import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from co_moe0 import PRIMARY, PRIMARY_REV
from fit_native_expert import CORPUS_REV, CORPUS_URL
from reciprocal_inference import GRANITE, GRANITE_REV, project
from replicate_granite_pair import execute, granite_expert

START, STOP, CONTEXT = 16384, 20480, 256
CONDITIONS = ("native", "pretrained", "zero", "random", "scrambled")
ROOT = Path(__file__).resolve().parent


class Settings(BaseModel):
    """The output destination is the only CLI setting; the protocol stays fixed."""

    output: Path = Path("results/substitution-benchmark")


def write_json(path, value):
    """Write this run's numerical record, rejecting nonfinite JSON numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.inference_mode()
def main():
    """Evaluate five frozen conditions once on a prospectively fixed text slice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Settings().output)
    settings = Settings.model_validate(vars(parser.parse_args()))
    # New destination required: historical runs are never overwritten.
    settings.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)
    protocol = {
        "experiment": "Fixed OLMoE/Granite native-expert substitution benchmark",
        "fixed_before_scoring_utc": datetime.now(UTC).isoformat(),
        "primary": {"model": PRIMARY, "revision": PRIMARY_REV, "layer": 7, "expert": 0},
        "donor": {"model": GRANITE, "revision": GRANITE_REV, "layer": 7, "expert": 0},
        "data": {
            "corpus": "Tiny Shakespeare",
            "revision": CORPUS_REV,
            "url": CORPUS_URL,
            "local_source": "results/native-expert-fit/corpus.txt",
            "token_range": [START, STOP],
            "selection": "Fixed unused Co-MoE corpus region; no selection by loss or routing",
            "context_tokens": CONTEXT,
            "context_count": (STOP - START) // CONTEXT,
            "context_resets": True,
            "input_tokens": STOP - START,
            "scored_targets": (STOP - START) // CONTEXT * (CONTEXT - 1),
            "scoring": "logits at position j predict reference token j+1; j=0..254 per context",
            "prior_fit_ranges": [[0, 8192], [336144, 340240]],
        },
        "conditions": list(CONDITIONS),
        "replacement": "Only layer-7 expert-0 raw output, before ORIGINAL mixture weight",
        "projections": "Truncate 2048 -> 1024; zero-pad 1024 -> 2048",
        "extra_alpha": None,
        "controls": {
            "random": {"distribution": "independent normal(0, 0.02)", "seed": 1},
            "scrambled": {"method": "Independent scalar permutations per gate/up/down", "seed": 2},
            "norm_matching": "Random/scrambled projected response per selected row matches the "
            "pretrained projected response on that same input, before mixture weighting",
            "norm_match_relative_tolerance": 1e-5,
        },
        "metrics": ["Mean next-token NLL (nats)", "exp(mean NLL)", "Paired NLL per context"],
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
            "One literary text slice, one fixed expert pair, one draw per weight control",
            "Unused in prior Co-MoE tests does not imply absence from model pretraining",
            "Context resets limit available history; not a standard full-corpus perplexity score",
            "Random/scrambled controls inherit the pretrained response norm, not its direction",
            "No statistical superiority, general capability, or performance-speed claim",
        ],
    }
    write_json(settings.output / "protocol.json", protocol)
    print("Protocol fixed before scoring; loading the existing pretrained expert.", flush=True)
    donor, _ = granite_expert()
    random, scrambled = copy.deepcopy(donor), copy.deepcopy(donor)
    rg, sg = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    for p in random.parameters():
        p.normal_(0, 0.02, generator=rg)
    for matrix in [
        scrambled.gate_up_proj[:, :512],
        scrambled.gate_up_proj[:, 512:],
        scrambled.down_proj,
    ]:
        flat = matrix.flatten()
        matrix.copy_(flat[torch.randperm(flat.numel(), generator=sg)].reshape_as(matrix))
    candidates = {"pretrained": donor, "random": random, "scrambled": scrambled}

    path = snapshot_download(
        PRIMARY, revision=PRIMARY_REV, allow_patterns=["*.json", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    corpus = (ROOT / protocol["data"]["local_source"]).read_text()
    all_ids = tokenizer(corpus, return_tensors="pt", verbose=False)["input_ids"]
    ids = all_ids[:, START:STOP].clone()
    if ids.numel() != STOP - START:
        raise ValueError("The frozen 4096-token corpus slice is unavailable")
    torch.save(ids, settings.output / "input_ids.pt")
    (settings.output / "slice.txt").write_text(
        tokenizer.decode(ids[0], clean_up_tokenization_spaces=False)
    )
    print("Loading the receiving backbone on CPU in float32.", flush=True)
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

    def capture(module, args):
        """Record native routing, including routes with a scored next-token target."""
        x, indices, weights = args
        positions = (indices == 0).any(dim=-1).nonzero().flatten()
        observed.update(
            indices=indices.clone(),
            weights=weights.clone(),
            positions=positions.clone(),
            selected_inputs=x[positions].clone(),
        )

    def capture_output(module, args, output):
        """Keep this context's MoE sum for unchanged-row checks against native."""
        observed["moe_output"] = output.clone()

    def replaced(module, hidden_states, indices, weights):
        """Reuse the native eager sum, replacing only expert 0's unweighted output."""
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(indices, num_classes=module.num_experts + 1).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            idx = expert_idx[0]
            if idx == module.num_experts:
                continue
            slots, positions = torch.where(mask[idx])
            x = hidden_states[positions]
            if idx == 0:
                if condition == "zero":
                    raw = torch.zeros_like(x)
                else:
                    raw = project(execute(candidates[condition], x[..., :1024]), 2048)
                    observed["donor_calls"] += 1
                    if condition in ("random", "scrambled"):
                        reference = project(execute(donor, x[..., :1024]), 2048)
                        observed["reference_calls"] += 1
                        target_norm = reference.norm(dim=-1, keepdim=True)
                        raw_norm = raw.norm(dim=-1, keepdim=True)
                        if not bool((target_norm > 0).all() and (raw_norm > 0).all()):
                            raise FloatingPointError("Cannot norm-match a zero/nonfinite response")
                        raw = raw * (target_norm / raw_norm)
                        error = (raw.norm(dim=-1, keepdim=True) - target_norm).abs() / target_norm
                        observed["norm_error"] = float(error.max())
                if not bool(torch.isfinite(raw).all()):
                    raise FloatingPointError(f"Nonfinite {condition} replacement output")
            else:
                gate, up = F.linear(x, module.gate_up_proj[idx]).chunk(2, -1)
                raw = F.linear(module.act_fn(gate) * up, module.down_proj[idx])
            final.index_add_(0, positions, (raw * weights[positions, slots, None]).to(final.dtype))
        return final

    handles = [
        experts.register_forward_pre_hook(capture),
        experts.register_forward_hook(capture_output),
    ]
    receipt = {"protocol": protocol, "outcome": "PARTIAL", "runs": runs}
    write_json(settings.output / "receipt.json", receipt)
    try:
        for condition in CONDITIONS:
            experts.forward = original if condition == "native" else MethodType(replaced, experts)
            windows, losses = [], []
            started = time.monotonic()
            for window, offset in enumerate(range(0, STOP - START, CONTEXT)):
                inputs = ids[:, offset : offset + CONTEXT]
                observed = {"donor_calls": 0, "reference_calls": 0, "norm_error": 0.0}
                output = model(input_ids=inputs, use_cache=False)
                if not bool(torch.isfinite(output.logits).all()):
                    raise FloatingPointError(f"Nonfinite logits: {condition}, context {window}")
                loss = F.cross_entropy(
                    output.logits[0, :-1], inputs[0, 1:], reduction="none"
                ).double()
                losses.append(loss.clone())
                positions = observed["positions"]
                untouched = torch.ones(CONTEXT, dtype=torch.bool)
                untouched[positions] = False
                if condition == "native":
                    baseline.append(observed)
                native = baseline[window]
                same_indices = torch.equal(observed["indices"], native["indices"])
                weight_error = float((observed["weights"] - native["weights"]).abs().max())
                input_error = (
                    float((observed["selected_inputs"] - native["selected_inputs"]).abs().max())
                    if positions.numel()
                    else 0.0
                )
                unselected_error = (
                    float(
                        (observed["moe_output"][untouched] - native["moe_output"][untouched])
                        .abs()
                        .max()
                    )
                    if bool(untouched.any())
                    else 0.0
                )
                if (
                    not same_indices
                    or weight_error != 0
                    or input_error != 0
                    or unselected_error != 0
                ):
                    raise RuntimeError(
                        f"Native slot invariants changed: {condition}, context {window}"
                    )
                if observed["norm_error"] >= 1e-5:
                    raise RuntimeError(f"Norm matching failed: {condition}, context {window}")
                nll = float(loss.mean())
                windows.append(
                    {
                        "context": window,
                        "corpus_range": [START + offset, START + offset + CONTEXT],
                        "scored_targets": loss.numel(),
                        "nll_sum": float(loss.sum()),
                        "mean_nll": nll,
                        "perplexity": math.exp(nll),
                        "delta_nll_vs_native": nll
                        - (
                            runs["native"]["windows"][window]["mean_nll"]
                            if condition != "native"
                            else nll
                        ),
                        "selected_input_rows": positions.numel(),
                        "selected_rows_with_scored_target": int((positions < CONTEXT - 1).sum()),
                        "selected_positions": positions.tolist(),
                        "donor_calls": observed["donor_calls"],
                        "norm_reference_calls": observed["reference_calls"],
                        "norm_match_max_relative_error": observed["norm_error"],
                        "routing_indices_equal": same_indices,
                        "routing_weights_max_abs": weight_error,
                        "selected_input_max_abs": input_error,
                        "unselected_moe_rows_max_abs": unselected_error,
                        "finite": bool(
                            torch.isfinite(loss).all()
                            and torch.isfinite(observed["moe_output"]).all()
                        ),
                    }
                )
                if (window + 1) % 4 == 0:
                    print(f"{condition}: {window + 1}/16 contexts completed", flush=True)
            scores[condition] = torch.stack(losses)
            mean_nll = float(scores[condition].mean())
            delta = mean_nll - (runs["native"]["mean_nll"] if condition != "native" else mean_nll)
            runs[condition] = {
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll),
                "delta_nll_vs_native": delta,
                "perplexity_change_percent_vs_native": 100 * math.expm1(delta),
                "scored_targets": scores[condition].numel(),
                "selected_input_rows": sum(w["selected_input_rows"] for w in windows),
                "selected_rows_with_scored_target": sum(
                    w["selected_rows_with_scored_target"] for w in windows
                ),
                "contexts_exercised": sum(w["selected_input_rows"] > 0 for w in windows),
                "finite": all(w["finite"] for w in windows),
                "seconds": round(time.monotonic() - started, 2),
                "windows": windows,
            }
            torch.save(scores, settings.output / "loss_per_token.pt")
            write_json(settings.output / "receipt.json", receipt)
            print(f"{condition}: NLL={mean_nll:.9f}, PPL={math.exp(mean_nll):.9f}", flush=True)
    finally:
        experts.forward = original
        for handle in handles:
            handle.remove()

    receipt["outcome"] = "COMPLETED"
    receipt["native_forward_method_restored"] = experts.forward == original
    receipt["paired_deltas_pretrained_minus_control"] = {
        name: {
            "aggregate_nll": runs["pretrained"]["mean_nll"] - runs[name]["mean_nll"],
            "per_context_nll": [
                a["mean_nll"] - b["mean_nll"]
                for a, b in zip(runs["pretrained"]["windows"], runs[name]["windows"], strict=True)
            ],
        }
        for name in ("native", "zero", "random", "scrambled")
    }
    torch.save(
        {
            "indices": torch.stack([b["indices"] for b in baseline]),
            "weights": torch.stack([b["weights"] for b in baseline]),
        },
        settings.output / "native_routing.pt",
    )
    write_json(settings.output / "receipt.json", receipt)
    print(f"COMPLETED: {settings.output / 'receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
