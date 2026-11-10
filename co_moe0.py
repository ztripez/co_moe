"""Run Co-MoE-0: OLMoE residual -> one pretrained DeepSeek-V2 expert -> OLMoE."""

import argparse
import ast
import copy
import json
import platform
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel, Field
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
from transformers.activations import ACT2FN

PRIMARY = "allenai/OLMoE-1B-7B-0924"
PRIMARY_REV = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
DONOR = "deepseek-ai/DeepSeek-V2-Lite"
DONOR_REV = "604d5664dddd88a0433dbae533b7fe9472482de0"


class Options(BaseModel):
    """The few controls needed for this fixed model pair."""

    weight_controls: bool = False
    alpha: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    tokens: int = Field(default=4, ge=1)
    threads: int = Field(default=12, ge=1)
    prompt: str = "The capital of France is"
    output: Path = Path("results/co-moe-0")


def fetch_range(url, start, end):
    """Read only the requested checkpoint bytes; reject a full-shard response."""
    with httpx.stream(
        "GET",
        url,
        headers={"Range": f"bytes={start}-{end}"},
        params={"co_moe_range": f"{start}-{end}"},
        follow_redirects=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"Checkpoint server ignored byte range: {response.status_code}")
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError("Incomplete checkpoint byte range")
    return data


def donor_expert():
    """Load three original matrices and only the publisher's standalone MLP class."""
    revision = DONOR_REV
    config_path = hf_hub_download(DONOR, "config.json", revision=revision)
    config = json.loads(Path(config_path).read_text())
    index_path = hf_hub_download(DONOR, "model.safetensors.index.json", revision=revision)
    index = json.loads(Path(index_path).read_text())["weight_map"]
    source = Path(hf_hub_download(DONOR, "modeling_deepseek.py", revision=revision)).read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MLP"
    )
    namespace = {"nn": torch.nn, "ACT2FN": ACT2FN}
    # Execute just this inspected, self-contained native expert class, never donor model code.
    exec(compile(ast.Module(body=[node], type_ignores=[]), "DeepseekV2MLP", "exec"), namespace)  # noqa: S102 - only the publisher's selected expert class
    expert = namespace["DeepseekV2MLP"](
        SimpleNamespace(**config), intermediate_size=config["moe_intermediate_size"]
    )
    cache = Path(config_path).parent / "co-moe-layer1-expert0.safetensors"
    prefix = "model.layers.1.mlp.experts.0."
    if not cache.exists():
        weights = {}
        headers = {}
        for name in expert.state_dict():
            key = prefix + name
            shard = index[key]
            url = f"https://huggingface.co/{DONOR}/resolve/{revision}/{shard}"
            if shard not in headers:
                length = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
                headers[shard] = (length, json.loads(fetch_range(url, 8, 7 + length)))
            length, header = headers[shard]
            entry = header[key]
            start, end = entry["data_offsets"]
            raw = fetch_range(url, 8 + length + start, 8 + length + end - 1)
            dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[
                entry["dtype"]
            ]
            weights[name] = (
                torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"]).clone()
            )
            print(f"Donor loaded {key}: {entry['shape']}", flush=True)
        save_file(weights, cache)
    expert.load_state_dict(load_file(cache))
    expert.eval().requires_grad_(False)
    return expert, revision, config


def norm(x):
    """Return a scalar L2 norm for the receipt."""
    return float(torch.linalg.vector_norm(x.float()))


def main():
    """Execute the five short inference paths and write their causal observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-controls", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    if args["output"] is None:
        args["output"] = Path(
            "results/weight-controls" if args["weight_controls"] else "results/co-moe-0"
        )
    opts = Options.model_validate(args)
    opts.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(opts.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    print("Loading primary on CPU in float32", flush=True)
    path = snapshot_download(
        PRIMARY, revision=PRIMARY_REV, allow_patterns=["*.json", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval().requires_grad_(False)
    inputs = tokenizer(opts.prompt, return_tensors="pt")["input_ids"]
    runs = {}
    tensors = {}
    expert = None
    reference_expert = None

    def run(name, alpha, sign=1):
        observed = {}
        calls = 0
        norm_errors = []

        def inject(module, args, output):
            nonlocal calls
            h = output[0] if isinstance(output, tuple) else output
            if "captured" not in observed:
                observed["captured"] = h[:, -1].clone()
            if alpha == 0:
                return output
            projected = h.roll(17, dims=-1) * sign
            foreign = expert(projected)
            contribution = foreign.roll(-17, dims=-1)
            if opts.weight_controls:
                # Match each token to the pretrained contribution on the SAME h.
                # No model output or native expert target enters this normalization.
                reference = reference_expert(projected).roll(-17, dims=-1)
                target_norm = reference.norm(dim=-1, keepdim=True)
                raw_norm = contribution.norm(dim=-1, keepdim=True)
                if not bool((raw_norm > 0).all() and (target_norm > 0).all()):
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
            changed = h + alpha * contribution
            calls += 1
            if "contribution" not in observed:
                observed["donor_input"] = projected[:, -1].clone()
                observed["donor_output"] = foreign[:, -1].clone()
                observed["contribution"] = contribution[:, -1].clone()
                observed["injected"] = changed[:, -1].clone()
                observed["shapes"] = [
                    list(h.shape),
                    list(projected.shape),
                    list(foreign.shape),
                    list(contribution.shape),
                ]
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        def downstream(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if "downstream" not in observed:
                observed["downstream"] = h[:, -1].clone()

        handles = [
            model.model.layers[7].register_forward_hook(inject),
            model.model.layers[8].register_forward_hook(downstream),
        ]
        started = time.time()
        generated = []
        ids = inputs
        cache = None
        finite = True
        try:
            for step in range(opts.tokens):
                out = model(input_ids=ids, past_key_values=cache, use_cache=True)
                finite = finite and bool(torch.isfinite(out.logits).all())
                if step == 0:
                    observed["logits"] = out.logits[:, -1].clone()
                ids = out.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(int(ids.item()))
                cache = out.past_key_values
        finally:
            for handle in handles:
                handle.remove()
        finite = finite and all(
            bool(torch.isfinite(v).all()) for v in observed.values() if isinstance(v, torch.Tensor)
        )
        runs[name] = {
            "alpha": alpha,
            "donor_input_sign": sign,
            "tokens": generated,
            "completion": tokenizer.decode(generated),
            "finite": finite,
            "expert_calls": calls,
            "seconds": round(time.time() - started, 2),
            "shapes": observed.get("shapes"),
            "norm_match_max_relative_error": max(norm_errors, default=0.0),
        }
        tensors[name] = {k: v for k, v in observed.items() if isinstance(v, torch.Tensor)}
        print(f"{name}: {runs[name]}", flush=True)

    run("baseline", 0)
    expert, donor_rev, donor_config = donor_expert()
    if opts.weight_controls:
        reference_expert = expert
        pretrained = expert
        randomized = copy.deepcopy(pretrained)
        scrambled = copy.deepcopy(pretrained)
        random_generator = torch.Generator().manual_seed(1)
        scramble_generator = torch.Generator().manual_seed(2)
        for parameter in randomized.parameters():
            parameter.normal_(
                mean=0, std=donor_config["initializer_range"], generator=random_generator
            )
        for parameter in scrambled.parameters():
            flat = parameter.flatten()
            parameter.copy_(
                flat[torch.randperm(flat.numel(), generator=scramble_generator)].reshape_as(
                    parameter
                )
            )
        for name, candidate in [
            ("pretrained", pretrained),
            ("random", randomized),
            ("scrambled", scrambled),
        ]:
            expert = candidate
            run(name, opts.alpha)
        b = tensors["baseline"]
        comparisons = {}
        for name in ["pretrained", "random", "scrambled"]:
            t = tensors[name]
            comparisons[name] = {
                "contribution_l2": norm(t["contribution"]),
                "injected_delta_l2": norm(t["injected"] - t["captured"]),
                "downstream_delta_l2": norm(t["downstream"] - b["downstream"]),
                "downstream_relative_delta_l2": norm(t["downstream"] - b["downstream"])
                / norm(b["downstream"]),
                "logits_delta_l2": norm(t["logits"] - b["logits"]),
                "logits_relative_delta_l2": norm(t["logits"] - b["logits"]) / norm(b["logits"]),
                "contribution_cosine_to_pretrained": float(
                    torch.nn.functional.cosine_similarity(
                        t["contribution"], tensors["pretrained"]["contribution"]
                    ).item()
                ),
                "logit_delta_cosine_to_pretrained": float(
                    torch.nn.functional.cosine_similarity(
                        t["logits"] - b["logits"], tensors["pretrained"]["logits"] - b["logits"]
                    ).item()
                ),
                "logits_l2_vs_pretrained": norm(t["logits"] - tensors["pretrained"]["logits"]),
                "captured_input_max_abs_vs_baseline": float(
                    (t["captured"] - b["captured"]).abs().max()
                ),
            }
        receipt = {
            "experiment": "Co-MoE-0 donor weight controls",
            "outcome": "OBSERVED" if all(r["finite"] for r in runs.values()) else "NONFINITE",
            "options": opts.model_dump(mode="json"),
            "primary": {"model": PRIMARY, "revision": PRIMARY_REV},
            "donor": {
                "model": DONOR,
                "revision": donor_rev,
                "expert": "model.layers.1.mlp.experts.0",
            },
            "random_control": {
                "seed": 1,
                "distribution": "independent normal(0, initializer_range)",
                "std": donor_config["initializer_range"],
            },
            "scrambled_control": {
                "seed": 2,
                "method": "independent permutation of all scalar entries within each original matrix",
            },
            "norm_matching": "Per token, after projection, before alpha: match pretrained output norm on the same receiving activation",
            "measurement": "Last prompt token, before generated histories can differ; layers 7 and 8, as in Co-MoE-0",
            "baseline_downstream_l2": norm(b["downstream"]),
            "baseline_logits_l2": norm(b["logits"]),
            "runtime": {
                "torch": torch.__version__,
                "transformers": __version__,
                "python": platform.python_version(),
                "device": "cpu",
                "dtype": "float32",
                "seed": 0,
            },
            "runs": runs,
            "comparisons": comparisons,
            "limitation": "One prompt and one random/scramble seed; distinguishability is not evidence of usefulness or a pretraining advantage.",
        }
        (opts.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        torch.save(tensors, opts.output / "activations.pt")
        print(json.dumps(receipt, indent=2), flush=True)
        return
    run("coupled", opts.alpha)
    run("coupled_repeat", opts.alpha)
    run("donor_input_negated", opts.alpha, -1)
    run("restored", 0)
    baseline, coupled, repeat, intervention, restored = [tensors[k] for k in runs]
    differences = {
        "coupled_downstream_l2": norm(coupled["downstream"] - baseline["downstream"]),
        "coupled_logits_l2": norm(coupled["logits"] - baseline["logits"]),
        "intervention_donor_output_l2": norm(
            intervention["donor_output"] - coupled["donor_output"]
        ),
        "intervention_contribution_l2": norm(
            intervention["contribution"] - coupled["contribution"]
        ),
        "intervention_downstream_l2": norm(intervention["downstream"] - coupled["downstream"]),
        "intervention_logits_l2": norm(intervention["logits"] - coupled["logits"]),
        "repeat_downstream_max_abs": float(
            (repeat["downstream"] - coupled["downstream"]).abs().max()
        ),
        "restored_downstream_max_abs": float(
            (restored["downstream"] - baseline["downstream"]).abs().max()
        ),
        "restored_logits_max_abs": float((restored["logits"] - baseline["logits"]).abs().max()),
    }
    passed = (
        all(r["finite"] for r in runs.values())
        and all(
            differences[k] > 0
            for k in [
                "coupled_downstream_l2",
                "intervention_donor_output_l2",
                "intervention_contribution_l2",
                "intervention_downstream_l2",
            ]
        )
        and all(
            differences[k] == 0
            for k in [
                "repeat_downstream_max_abs",
                "restored_downstream_max_abs",
                "restored_logits_max_abs",
            ]
        )
        and runs["baseline"]["tokens"] == runs["restored"]["tokens"]
    )
    receipt = {
        "outcome": "PASS" if passed else "NEGATIVE RESULT",
        "options": opts.model_dump(mode="json"),
        "primary": {"model": PRIMARY, "revision": PRIMARY_REV},
        "donor": {
            "model": DONOR,
            "revision": donor_rev,
            "expert": "model.layers.1.mlp.experts.0",
            "hidden_size": donor_config["hidden_size"],
            "intermediate_size": donor_config["moe_intermediate_size"],
        },
        "injection": "layer 7 output, after attention and native MoE residual additions; observed layer 8 output, last prompt token",
        "projections": "2048 -> 2048 cyclic permutation +17; inverse -17 on output; no fitted parameters",
        "donor_backbone_instantiated_or_executed": False,
        "runtime": {
            "torch": torch.__version__,
            "transformers": __version__,
            "python": platform.python_version(),
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
            "restoration_tolerance": 0,
        },
        "runs": runs,
        "differences": differences,
    }
    (opts.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(tensors, opts.output / "activations.pt")
    print(json.dumps({"outcome": receipt["outcome"], **differences}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Co-MoE-0 criteria were not all demonstrated; see receipt")


if __name__ == "__main__":
    main()
