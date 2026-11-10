"""Replace one live routed OLMoE expert with saved foreign adapters or controls."""

import argparse
import json
import platform
import time
from pathlib import Path
from types import MethodType

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from co_moe0 import DONOR, DONOR_REV, PRIMARY, PRIMARY_REV, donor_expert
from fit_native_expert import ForeignFit


class Settings(BaseModel):
    """Saved fit location and short generation controls for this one expert."""

    fit: Path = Path("results/native-expert-fit")
    output: Path = Path("results/live-substitution")
    tokens: int = Field(default=8, ge=1)
    threads: int = Field(default=12, ge=1)


def main():
    """Run native, foreign, bypass, zero, and restored live inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=Path("results/native-expert-fit"))
    parser.add_argument("--output", type=Path, default=Path("results/live-substitution"))
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=12)
    settings = Settings.model_validate(vars(parser.parse_args()))
    settings.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    fit_receipt = json.loads((settings.fit / "receipt.json").read_text())
    saved = torch.load(settings.fit / "adapters.pt", map_location="cpu", weights_only=True)
    donor, _, _ = donor_expert()
    foreign = ForeignFit(donor, saved["pretrained"]["expert_rms"])
    for name, parameter in foreign.named_parameters():
        if not name.startswith("expert."):
            parameter.copy_(saved["pretrained"]["parameters"][name])
    foreign.eval().requires_grad_(False)
    bypass = nn.Sequential(
        nn.Linear(2048, 33, bias=False), nn.SiLU(), nn.Linear(33, 2048, bias=False)
    )
    bypass.load_state_dict(saved["bypass"]["parameters"])
    bypass.eval().requires_grad_(False)
    target_rms = saved["pretrained"]["target_rms"]
    path = snapshot_download(
        PRIMARY, revision=PRIMARY_REV, allow_patterns=["*.json", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    corpus_ids = tokenizer(
        (settings.fit / "corpus.txt").read_text(), return_tensors="pt", verbose=False
    )["input_ids"]
    start, end = fit_receipt["data"]["heldout_token_range"]
    heldout = corpus_ids[:, start:end]
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
        )
        .eval()
        .requires_grad_(False)
    )
    experts = model.model.layers[7].mlp.experts
    original_forward = experts.forward
    records, tensors = {}, {}

    def run(name, prompt):
        observed = {}
        total_selections = 0
        replacement_calls = 0
        replacement_rows = 0
        all_finite = True
        first_forward = True

        def routes(module, args):
            nonlocal total_selections
            x, indices, weights = args
            positions, slots = torch.where(indices == 0)
            total_selections += len(positions)
            if first_forward:
                observed["router_indices"] = indices.clone()
                observed["router_weights"] = weights.clone()
                observed["selected_positions"] = positions.clone()
                observed["expert_input"] = x[positions].clone()
                observed["selected_weights"] = weights[positions, slots].clone()
                if name in {"native", "restored"} and len(positions):
                    gate, up = F.linear(x[positions], module.gate_up_proj[0]).chunk(2, -1)
                    observed["raw_expert_output"] = F.linear(
                        module.act_fn(gate) * up, module.down_proj[0]
                    ).clone()

        def replacement_forward(module, hidden_states, top_k_index, top_k_weights):
            nonlocal replacement_calls, replacement_rows, all_finite
            # Same native expert iteration, accumulation and mixture weights.
            # Only expert 0's raw output branch changes. The router is untouched.
            final_hidden_states = torch.zeros_like(hidden_states)
            expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts + 1).permute(
                2, 1, 0
            )
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == module.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                if expert_idx == 0:
                    if name == "foreign":
                        raw = target_rms * foreign(current_state)
                    elif name == "bypass":
                        raw = saved["bypass"]["target_rms"] * bypass(current_state)
                    else:
                        raw = torch.zeros_like(current_state)
                    replacement_calls += 1
                    replacement_rows += len(token_idx)
                    all_finite = all_finite and bool(torch.isfinite(raw).all())
                    if first_forward:
                        observed["raw_expert_output"] = raw[torch.argsort(token_idx)].clone()
                else:
                    gate, up = F.linear(current_state, module.gate_up_proj[expert_idx]).chunk(2, -1)
                    raw = F.linear(module.act_fn(gate) * up, module.down_proj[expert_idx])
                weighted = raw * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
            return final_hidden_states

        def moe_output(module, args, output):
            nonlocal all_finite
            all_finite = all_finite and bool(torch.isfinite(output).all())
            if first_forward:
                observed["moe_output"] = output.clone()

        def downstream(module, args, output):
            nonlocal all_finite
            h = output[0] if isinstance(output, tuple) else output
            all_finite = all_finite and bool(torch.isfinite(h).all())
            if first_forward:
                observed["downstream"] = h.clone()

        if name not in {"native", "restored"}:
            experts.forward = MethodType(replacement_forward, experts)
        handles = [
            experts.register_forward_pre_hook(routes),
            experts.register_forward_hook(moe_output),
            model.model.layers[8].register_forward_hook(downstream),
        ]
        generated, cache = [], None
        ids = prompt
        started = time.time()
        try:
            for _ in range(settings.tokens):
                out = model(input_ids=ids, past_key_values=cache, use_cache=True)
                all_finite = all_finite and bool(torch.isfinite(out.logits).all())
                if first_forward:
                    observed["logits"] = out.logits[:, -1].clone()
                first_forward = False
                ids = out.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(int(ids.item()))
                cache = out.past_key_values
        finally:
            experts.forward = original_forward
            for handle in handles:
                handle.remove()
        tensors[name] = observed
        records[name] = {
            "generated_tokens": generated,
            "completion": tokenizer.decode(generated, clean_up_tokenization_spaces=False),
            "finite": all_finite,
            "prefill_expert0_selections": len(observed["selected_positions"]),
            "total_expert0_selections": total_selections,
            "replacement_calls": replacement_calls,
            "replacement_rows": replacement_rows,
            "seconds": round(time.time() - started, 2),
        }
        print(f"{name}: {json.dumps(records[name])}", flush=True)

    # Select only for actual native routing, never for substitution outcomes.
    chunk_size = fit_receipt["data"]["context_tokens"]
    for offset in range(0, heldout.shape[1], chunk_size):
        prompt = heldout[:, offset : offset + chunk_size]
        run("native", prompt)
        if records["native"]["prefill_expert0_selections"] > 0:
            break
    else:
        raise RuntimeError("No held-out chunk selected native expert 0")
    for name in ["foreign", "bypass", "zero", "restored"]:
        run(name, prompt)
    baseline = tensors["native"]
    positions = baseline["selected_positions"]
    untouched = torch.ones(prompt.numel(), dtype=torch.bool)
    untouched[positions] = False
    comparisons = {}
    for name in ["foreign", "bypass", "zero", "restored"]:
        t = tensors[name]
        raw_delta = t["raw_expert_output"] - baseline["raw_expert_output"]
        weighted_delta = raw_delta * baseline["selected_weights"][:, None]
        actual_delta = t["moe_output"][positions] - baseline["moe_output"][positions]
        comparisons[name] = {
            "prefill_router_indices_equal": torch.equal(
                t["router_indices"], baseline["router_indices"]
            ),
            "prefill_router_weights_max_abs": float(
                (t["router_weights"] - baseline["router_weights"]).abs().max()
            ),
            "raw_expert_output_delta_l2": float(raw_delta.norm()),
            "moe_output_delta_l2": float((t["moe_output"] - baseline["moe_output"]).norm()),
            "mixture_weight_application_max_abs_error": float(
                (actual_delta - weighted_delta).abs().max()
            ),
            "unselected_moe_rows_max_abs": float(
                (t["moe_output"][untouched] - baseline["moe_output"][untouched]).abs().max()
            )
            if bool(untouched.any())
            else 0,
            "downstream_selected_positions_delta_l2": float(
                (t["downstream"][:, positions] - baseline["downstream"][:, positions]).norm()
            ),
            "downstream_max_abs": float((t["downstream"] - baseline["downstream"]).abs().max()),
            "last_prompt_logits_delta_l2": float((t["logits"] - baseline["logits"]).norm()),
            "last_prompt_logits_max_abs": float((t["logits"] - baseline["logits"]).abs().max()),
            "generated_tokens_equal": records[name]["generated_tokens"]
            == records["native"]["generated_tokens"],
        }
    passed = (
        all(v["finite"] for v in records.values())
        and all(records[n]["replacement_rows"] > 0 for n in ["foreign", "bypass", "zero"])
        and all(
            c["prefill_router_indices_equal"]
            and c["prefill_router_weights_max_abs"] == 0
            and c["unselected_moe_rows_max_abs"] == 0
            for c in comparisons.values()
        )
        and comparisons["foreign"]["downstream_selected_positions_delta_l2"] > 0
        and comparisons["restored"]["downstream_max_abs"] == 0
        and comparisons["restored"]["last_prompt_logits_max_abs"] == 0
        and comparisons["restored"]["generated_tokens_equal"]
    )
    receipt = {
        "experiment": "Live native-expert substitution",
        "outcome": "PASS" if passed else "PARTIAL",
        "settings": settings.model_dump(mode="json"),
        "primary": {
            "model": PRIMARY,
            "revision": PRIMARY_REV,
            "expert": "model.layers.7.mlp.experts.0",
        },
        "donor": {"model": DONOR, "revision": DONOR_REV, "expert": "model.layers.1.mlp.experts.0"},
        "adapter_checkpoint": str(settings.fit / "adapters.pt"),
        "adapter_step": fit_receipt["settings"]["steps"],
        "checkpoint_choice": "Previously saved final step; no additional training or checkpoint selection",
        "prompt": {
            "corpus_range": [start + offset, start + offset + prompt.numel()],
            "token_ids": prompt.flatten().tolist(),
            "text": tokenizer.decode(prompt.flatten(), clean_up_tokenization_spaces=False),
            "selection": "First previously held-out context chunk whose native prefill routes to expert 0",
        },
        "replacement": "Replace only expert 0 raw output; retain original top-k routing, original mixture weights, and all other native expert computations. No additive augmentation or norm matching.",
        "observation": "Same prompt prefill: raw target expert, native weighted MoE output, layer 8 output at selected positions, and last-prompt logits. Generation then runs freely with native routing.",
        "restoration_tolerance": 0,
        "donor_backbone_executed": False,
        "runs": records,
        "comparisons": comparisons,
        "runtime": {
            "torch": torch.__version__,
            "transformers": __version__,
            "python": platform.python_version(),
            "device": "cpu",
            "experts_implementation": "eager",
            "dtype": "float32",
            "seed": 0,
        },
        "limitation": "Execution substitution only; no claim of useful learned transfer, native functional equivalence, or output quality.",
    }
    (settings.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(tensors, settings.output / "activations.pt")
    print(
        json.dumps({"outcome": receipt["outcome"], "comparisons": comparisons}, indent=2),
        flush=True,
    )
    if not passed:
        raise RuntimeError("Live substitution criteria not all demonstrated; see receipt")


if __name__ == "__main__":
    main()
