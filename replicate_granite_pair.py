"""Replicate injection, weight controls, and routed substitution with OLMoE/Granite."""

import argparse
import copy
import json
import platform
from pathlib import Path
from types import MethodType

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts

from co_moe0 import PRIMARY, PRIMARY_REV
from reciprocal_inference import GRANITE, GRANITE_REV, project


class Settings(BaseModel):
    """Only the output destination varies; experimental settings remain fixed."""

    output: Path = Path("results/granite-pair-replication")


def granite_expert():
    """Load only expert 0 slices, preserving the publisher's native tensor layout."""
    path = Path(
        snapshot_download(GRANITE, revision=GRANITE_REV, allow_patterns=["*.json", "*.safetensors"])
    )
    config = AutoConfig.from_pretrained(path)
    config.num_local_experts = 1
    config._experts_implementation = "eager"
    expert = GraniteMoeExperts(config)
    keys = {
        "model.layers.7.block_sparse_moe.input_linear.weight": "gate_up_proj",
        "model.layers.7.block_sparse_moe.output_linear.weight": "down_proj",
    }
    weights = {}
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key, target in keys.items():
                if key in source.keys():  # noqa: SIM118 - safe_open exposes keys(), not dict containment
                    weights[target] = source.get_slice(key)[0:1].clone().float()
    # These two names are the exact rename-only Granite mappings used by
    # Transformers. No weight transposition, reordering or reinitialization.
    expert.load_state_dict(weights)
    return expert.eval().requires_grad_(False), config


def execute(expert, x):
    """Evaluate the selected native expert with unit routing weight."""
    flat = x.reshape(-1, 1024)
    y = expert.forward(
        flat, torch.zeros((len(flat), 1), dtype=torch.long), torch.ones((len(flat), 1))
    )
    return y.reshape_as(x)


def main():
    """Run R0-R2, reference R3, and write new receipts without touching history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/granite-pair-replication"))
    settings = Settings.model_validate(vars(parser.parse_args()))
    settings.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    reciprocal = json.loads(Path("results/reciprocal-inference/receipt.json").read_text())
    feedback = json.loads(Path("results/feedback-inference/receipt.json").read_text())
    if (
        reciprocal["models"]["A"]["revision"] != PRIMARY_REV
        or reciprocal["models"]["B"]["revision"] != GRANITE_REV
    ):
        raise RuntimeError("Replication constants differ from the pinned reciprocal revisions")
    donor, donor_config = granite_expert()
    random = copy.deepcopy(donor)
    scrambled = copy.deepcopy(donor)
    rg, sg = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    for p in random.parameters():
        p.normal_(0, donor_config.initializer_range, generator=rg)
    # Preserve separate gate/up/down empirical distributions, as in the
    # original three-matrix DeepSeek scrambling control.
    for matrix in [
        scrambled.gate_up_proj[:, :512],
        scrambled.gate_up_proj[:, 512:],
        scrambled.down_proj,
    ]:
        flat = matrix.flatten()
        matrix.copy_(flat[torch.randperm(flat.numel(), generator=sg)].reshape_as(matrix))
    path = snapshot_download(
        PRIMARY, revision=PRIMARY_REV, allow_patterns=["*.json", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
        )
        .eval()
        .requires_grad_(False)
    )
    short_prompt = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    source_context = json.loads(Path("results/live-substitution/receipt.json").read_text())[
        "prompt"
    ]
    routed_prompt = torch.tensor([source_context["token_ids"]], dtype=torch.long)
    records, tensors = {}, {}

    def generate(label, prompt, tokens, observed, finite, calls):
        """Complete one hooked inference and retain its fixed-position evidence."""
        generated, cache, ids = [], None, prompt
        for step in range(tokens):
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
            finite.append(bool(torch.isfinite(out.logits).all()))
            if step == 0:
                observed["logits"] = out.logits[:, -1].clone()
            ids = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(ids.item()))
            cache = out.past_key_values
        tensors[label] = observed
        records[label] = {
            "tokens": generated,
            "completion": tokenizer.decode(generated, clean_up_tokenization_spaces=False),
            "finite": all(finite) and all(bool(torch.isfinite(v).all()) for v in observed.values()),
            "donor_calls": calls[0],
        }
        print(f"{label}: {json.dumps(records[label])}", flush=True)

    def inject_run(label, candidate=None, sign=1, match=False):
        observed, finite, calls, norm_errors = {}, [], [0], []

        def inject(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            first = "captured" not in observed
            if first:
                observed["captured"] = h[:, -1].clone()
            if candidate is None:
                return output
            x = h[..., :1024] * sign
            y = execute(candidate, x)
            contribution = project(y, 2048)
            if match:
                reference = project(execute(donor, x), 2048)
                target_norm, raw_norm = (
                    reference.norm(dim=-1, keepdim=True),
                    contribution.norm(dim=-1, keepdim=True),
                )
                if not bool((target_norm > 0).all() and (raw_norm > 0).all()):
                    raise RuntimeError("Cannot norm-match a zero contribution")
                contribution = contribution * (target_norm / raw_norm)
                norm_errors.append(
                    float(
                        (
                            (contribution.norm(dim=-1, keepdim=True) - target_norm).abs()
                            / target_norm
                        ).max()
                    )
                )
            changed = h + 0.01 * contribution
            calls[0] += 1
            finite.extend([bool(torch.isfinite(y).all()), bool(torch.isfinite(changed).all())])
            if first:
                observed.update(
                    {
                        "donor_input": x[:, -1].clone(),
                        "donor_output": y[:, -1].clone(),
                        "contribution": contribution[:, -1].clone(),
                        "injected": changed[:, -1].clone(),
                    }
                )
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        def downstream(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            finite.append(bool(torch.isfinite(h).all()))
            if "downstream" not in observed:
                observed["downstream"] = h[:, -1].clone()

        handles = [
            model.model.layers[7].register_forward_hook(inject),
            model.model.layers[8].register_forward_hook(downstream),
        ]
        try:
            generate(label, short_prompt, 4, observed, finite, calls)
        finally:
            for handle in handles:
                handle.remove()
        records[label]["norm_match_max_relative_error"] = max(norm_errors, default=0.0)

    inject_run("R0_baseline")
    inject_run("R0_pretrained", donor)
    inject_run("R0_input_negated", donor, -1)
    inject_run("R0_restored")
    # The pretrained contribution is its own norm target (scale exactly 1),
    # so this already completed run is the R1 pretrained condition as well.
    inject_run("R1_random", random, match=True)
    inject_run("R1_scrambled", scrambled, match=True)

    experts = model.model.layers[7].mlp.experts
    original_forward = experts.forward

    def substitute_run(label, replacement=False, zero=False):
        observed, finite, calls = {}, [], [0]
        total_rows = 0

        def routes(module, args):
            nonlocal total_rows
            x, indices, weights = args
            positions, slots = torch.where(indices == 0)
            total_rows += len(positions)
            if "router_indices" not in observed:
                observed.update(
                    {
                        "router_indices": indices.clone(),
                        "router_weights": weights.clone(),
                        "positions": positions.clone(),
                        "selected_weights": weights[positions, slots].clone(),
                        "expert_input": x[positions].clone(),
                    }
                )
                if not replacement and len(positions):
                    gate, up = F.linear(x[positions], module.gate_up_proj[0]).chunk(2, -1)
                    observed["raw_output"] = F.linear(
                        module.act_fn(gate) * up, module.down_proj[0]
                    ).clone()

        def replaced(module, hidden_states, indices, weights):
            final = torch.zeros_like(hidden_states)
            mask = F.one_hot(indices, num_classes=module.num_experts + 1).permute(2, 1, 0)
            for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
                idx = expert_idx[0]
                if idx == module.num_experts:
                    continue
                slots, positions = torch.where(mask[idx])
                x = hidden_states[positions]
                if idx == 0:
                    raw = (
                        torch.zeros_like(x)
                        if zero
                        else project(execute(donor, x[..., :1024]), 2048)
                    )
                    if not zero:
                        calls[0] += 1
                    if "raw_output" not in observed:
                        observed["raw_output"] = raw[torch.argsort(positions)].clone()
                else:
                    gate, up = F.linear(x, module.gate_up_proj[idx]).chunk(2, -1)
                    raw = F.linear(module.act_fn(gate) * up, module.down_proj[idx])
                final.index_add_(
                    0, positions, (raw * weights[positions, slots, None]).to(final.dtype)
                )
            return final

        def moe_output(module, args, output):
            finite.append(bool(torch.isfinite(output).all()))
            if "moe_output" not in observed:
                observed["moe_output"] = output.clone()

        def downstream(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            finite.append(bool(torch.isfinite(h).all()))
            if "downstream" not in observed:
                observed["downstream"] = h.clone()

        handles = [
            experts.register_forward_pre_hook(routes),
            experts.register_forward_hook(moe_output),
            model.model.layers[8].register_forward_hook(downstream),
        ]
        if replacement:
            experts.forward = MethodType(replaced, experts)
        try:
            generate(label, routed_prompt, 8, observed, finite, calls)
        finally:
            experts.forward = original_forward
            for handle in handles:
                handle.remove()
        records[label].update(
            {
                "prefill_selected_rows": len(observed["positions"]),
                "total_selected_rows": total_rows,
                "replacement_exercised": replacement and total_rows > 0,
            }
        )

    substitute_run("R2_native")
    substitute_run("R2_foreign", replacement=True)
    substitute_run("R2_zero", replacement=True, zero=True)
    substitute_run("R2_restored")

    def l2(a, b):
        return float((a - b).norm())

    def exact_restore(native, restored):
        b, r = tensors[native], tensors[restored]
        return {
            "downstream_max_abs": float((b["downstream"] - r["downstream"]).abs().max()),
            "logits_max_abs": float((b["logits"] - r["logits"]).abs().max()),
            "tokens_equal": records[native]["tokens"] == records[restored]["tokens"],
        }

    b, p, i = [tensors[k] for k in ["R0_baseline", "R0_pretrained", "R0_input_negated"]]
    r0 = {
        "coupled_downstream_l2": l2(p["downstream"], b["downstream"]),
        "coupled_logits_l2": l2(p["logits"], b["logits"]),
        "intervention_donor_output_l2": l2(i["donor_output"], p["donor_output"]),
        "intervention_contribution_l2": l2(i["contribution"], p["contribution"]),
        "intervention_downstream_l2": l2(i["downstream"], p["downstream"]),
        "intervention_logits_l2": l2(i["logits"], p["logits"]),
        "restoration": exact_restore("R0_baseline", "R0_restored"),
    }
    r1 = {}
    for name, label in [
        ("pretrained", "R0_pretrained"),
        ("random", "R1_random"),
        ("scrambled", "R1_scrambled"),
    ]:
        t = tensors[label]
        r1[name] = {
            "run": label,
            "contribution_l2": float(t["contribution"].norm()),
            "downstream_l2_vs_native": l2(t["downstream"], b["downstream"]),
            "logits_l2_vs_native": l2(t["logits"], b["logits"]),
            "contribution_cosine_to_pretrained": float(
                F.cosine_similarity(t["contribution"], p["contribution"]).item()
            ),
            "logits_l2_vs_pretrained": l2(t["logits"], p["logits"]),
            "input_max_abs_vs_pretrained": float((t["captured"] - p["captured"]).abs().max()),
            "norm_match_max_relative_error": records[label]["norm_match_max_relative_error"],
        }
    native = tensors["R2_native"]
    selected = native["positions"]
    untouched = torch.ones(routed_prompt.numel(), dtype=torch.bool)
    untouched[selected] = False
    r2 = {}
    for label in ["R2_foreign", "R2_zero", "R2_restored"]:
        t = tensors[label]
        raw_delta = t["raw_output"] - native["raw_output"]
        weighted_delta = raw_delta * native["selected_weights"][:, None]
        actual_delta = t["moe_output"][selected] - native["moe_output"][selected]
        r2[label] = {
            "router_indices_equal": torch.equal(t["router_indices"], native["router_indices"]),
            "router_weights_max_abs": float(
                (t["router_weights"] - native["router_weights"]).abs().max()
            ),
            "unselected_moe_rows_max_abs": float(
                (t["moe_output"][untouched] - native["moe_output"][untouched]).abs().max()
            ),
            "mixture_weight_application_max_abs_error": float(
                (actual_delta - weighted_delta).abs().max()
            ),
            "raw_output_delta_l2": float(raw_delta.norm()),
            "downstream_selected_l2": l2(
                t["downstream"][:, selected], native["downstream"][:, selected]
            ),
            "logits_delta_l2": l2(t["logits"], native["logits"]),
        }
    r2["foreign_vs_zero_logits_l2"] = l2(
        tensors["R2_foreign"]["logits"], tensors["R2_zero"]["logits"]
    )
    r2["restoration"] = exact_restore("R2_native", "R2_restored")

    def restored_ok(r):
        return r["downstream_max_abs"] == 0 and r["logits_max_abs"] == 0 and r["tokens_equal"]

    statuses = {
        "R0": "PASS"
        if all(
            records[k]["finite"]
            for k in ["R0_baseline", "R0_pretrained", "R0_input_negated", "R0_restored"]
        )
        and all(
            r0[k] > 0
            for k in [
                "coupled_downstream_l2",
                "intervention_donor_output_l2",
                "intervention_contribution_l2",
                "intervention_downstream_l2",
            ]
        )
        and restored_ok(r0["restoration"])
        else "PARTIAL",
        "R1": "PASS"
        if all(
            records[k]["finite"]
            for k in ["R0_baseline", "R0_pretrained", "R1_random", "R1_scrambled"]
        )
        and all(
            v["input_max_abs_vs_pretrained"] == 0
            and v["contribution_l2"] > 0
            and v["norm_match_max_relative_error"] < 1e-5
            for v in r1.values()
        )
        else "PARTIAL",
        "R2": "PASS"
        if all(records[k]["finite"] for k in ["R2_native", "R2_foreign", "R2_zero", "R2_restored"])
        and records["R2_foreign"]["replacement_exercised"]
        and records["R2_foreign"]["donor_calls"] > 0
        and all(
            r2[k]["router_indices_equal"]
            and r2[k]["router_weights_max_abs"] == 0
            and r2[k]["unselected_moe_rows_max_abs"] == 0
            for k in ["R2_foreign", "R2_zero", "R2_restored"]
        )
        and r2["R2_foreign"]["downstream_selected_l2"] > 0
        and restored_ok(r2["restoration"])
        else "PARTIAL",
    }
    receipt = {
        "experiment": "Consistent OLMoE/Granite pair replication",
        "statuses": statuses,
        "primary": {"model": PRIMARY, "revision": PRIMARY_REV, "expert": "layer 7 expert 0"},
        "donor": {
            "model": GRANITE,
            "revision": GRANITE_REV,
            "expert": "layer 7 expert 0",
            "input_size": 1024,
            "intermediate_size": 512,
        },
        "projections": "Input truncation 2048 -> 1024; output zero padding 1024 -> 2048. No permutation or learned maps.",
        "boundaries": {
            "R0_R1": "Add alpha 0.01 * projected donor response at layer-7 residual output; observe layer 8, last prompt token",
            "R2": "Replace only layer-7 expert-0 raw output before ORIGINAL router mixture weight; no extra alpha or norm matching. Observe layer 8 at selected prompt positions.",
        },
        "controls": {
            "random_seed": 1,
            "random_distribution": "independent normal(0, 0.02)",
            "scramble_seed": 2,
            "scramble_method": "Independent scalar permutations within each original gate/up/down matrix",
            "norm_matching": "Per token, after projection, before alpha, to pretrained response on the same activation; relative tolerance 1e-5",
            "pretrained_reuse": "R0 active serves as R1 pretrained: the norm target is itself, scale exactly 1",
        },
        "inputs": {
            "R0_R1": {
                "prompt": "The capital of France is",
                "tokens": short_prompt.flatten().tolist(),
                "generation_steps": 4,
            },
            "R2": {
                "source": "results/live-substitution/receipt.json",
                "prompt": source_context,
                "generation_steps": 8,
            },
        },
        "R0": r0,
        "R1": r1,
        "R2": r2,
        "R3": {
            "rerun": False,
            "reciprocal": {
                "receipt": "results/reciprocal-inference/receipt.json",
                "outcome": reciprocal["outcome"],
            },
            "feedback": {
                "receipt": "results/feedback-inference/receipt.json",
                "outcome": feedback["outcome"],
            },
            "implementation_changed": False,
        },
        "runs": records,
        "training": "None; no learned adapters loaded",
        "donor_backbone_instantiated_or_executed": False,
        "runtime": {
            "torch": torch.__version__,
            "transformers": __version__,
            "python": platform.python_version(),
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
            "threads": 12,
            "attention": "eager",
            "experts": "eager",
        },
        "limitations": "Execution and causal replication only; no donor quality ranking or learned-transfer advantage. Historical DeepSeek files are not written by this program.",
    }
    (settings.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(tensors, settings.output / "activations.pt")
    print(json.dumps({"statuses": statuses, "R0": r0, "R1": r1, "R2": r2}, indent=2), flush=True)
    if any(s != "PASS" for s in statuses.values()):
        raise RuntimeError("Replication is incomplete; see statuses in receipt")


if __name__ == "__main__":
    main()
