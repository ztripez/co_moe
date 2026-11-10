"""One causally ordered A -> B -> A expert-response loop in live MoE inference."""

import argparse
import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from reciprocal_inference import SPECS, project


class Settings(BaseModel):
    """Fixed-pair controls, with the same coefficient and prompts as reciprocity."""

    output: Path = Path("results/feedback-inference")
    tokens: int = Field(default=4, ge=1)
    alpha: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    threads: int = Field(default=6, ge=1)


def main():
    """Execute one two-round loop, suppress its first message, restore, and stop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/feedback-inference"))
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=6)
    settings = Settings.model_validate(vars(parser.parse_args()))
    settings.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    models, tokenizers, prompts = {}, {}, {}
    for key, (repo, revision, prompt) in SPECS.items():
        print(f"Loading live backbone {key}: {repo}", flush=True)
        path = snapshot_download(
            repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]
        )
        tokenizers[key] = AutoTokenizer.from_pretrained(path)
        prompts[key] = tokenizers[key](prompt, return_tensors="pt")["input_ids"]
        models[key] = (
            AutoModelForCausalLM.from_pretrained(
                path,
                dtype=torch.float32,
                attn_implementation="eager",
                experts_implementation="eager",
            )
            .eval()
            .requires_grad_(False)
        )
    moes = {
        "A": {i: models["A"].model.layers[i].mlp for i in (7, 9)},
        "B": {i: models["B"].model.layers[i].block_sparse_moe for i in (7, 9)},
    }
    runs, tensors = {}, {}

    def paired_run(name, active, suppress_first=False):
        barriers = [threading.Barrier(2), threading.Barrier(2)]
        messages, observations, records, trace = {}, {}, {}, []

        def stream(key):
            observed = {}
            first = True
            finite = True
            extra_expert_calls = 0
            rounds = 0

            def install_round(round_index, layer_index):
                layer = models[key].model.layers[layer_index]
                label = f"round{round_index + 1}"

                def capture_input(module, args):
                    if first:
                        observed[f"{label}_expert_input"] = args[0][:, -1].clone()

                def exchange(module, args, output):
                    nonlocal finite, extra_expert_calls, rounds
                    if not first:
                        return output
                    h = output[0] if isinstance(output, tuple) else output
                    observed[f"{label}_boundary_before"] = h[:, -1].clone()
                    trace.append({"stream": key, "event": f"{label}_boundary_ready"})
                    sender = (round_index == 0 and key == "A") or (round_index == 1 and key == "B")
                    if active and sender:
                        # Reuse the SAME layer-7 expert 0 for both experiments.
                        # B's return input is its actual later layer-9 MoE input,
                        # which is computed only after receiving the first message.
                        x = observed[f"{label}_expert_input"]
                        response = moes[key][7].experts.forward(
                            x, torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1))
                        )
                        extra_expert_calls += 1
                        observed[f"{label}_response"] = response.clone()
                        delivered = (
                            torch.zeros_like(response)
                            if round_index == 0 and suppress_first
                            else response.clone()
                        )
                        messages[round_index] = delivered
                        observed[f"{label}_sent"] = delivered.clone()
                        trace.append({"stream": key, "event": f"{label}_message_ready"})
                    barriers[round_index].wait(timeout=120)
                    rounds += 1
                    if active and not sender:
                        received = messages[round_index]
                        contribution = project(received, h.shape[-1])
                        changed = h.clone()
                        changed[:, -1] += settings.alpha * contribution
                        observed[f"{label}_received"] = received.clone()
                        observed[f"{label}_contribution"] = contribution.clone()
                        observed[f"{label}_boundary_after"] = changed[:, -1].clone()
                        finite = finite and bool(torch.isfinite(changed).all())
                        trace.append({"stream": key, "event": f"{label}_message_injected"})
                        return (changed, *output[1:]) if isinstance(output, tuple) else changed
                    trace.append({"stream": key, "event": f"{label}_resume_without_injection"})
                    return output

                return [
                    moes[key][layer_index].register_forward_pre_hook(capture_input),
                    layer.register_forward_hook(exchange),
                ]

            def downstream(module, args, output):
                nonlocal finite
                h = output[0] if isinstance(output, tuple) else output
                finite = finite and bool(torch.isfinite(h).all())
                if first:
                    observed["downstream_layer10"] = h[:, -1].clone()

            handles = install_round(0, 7) + install_round(1, 9)
            handles.append(models[key].model.layers[10].register_forward_hook(downstream))
            ids, cache, generated = prompts[key], None, []
            started = time.time()
            try:
                with torch.inference_mode():
                    for _ in range(settings.tokens):
                        out = models[key](input_ids=ids, past_key_values=cache, use_cache=True)
                        finite = finite and bool(torch.isfinite(out.logits).all())
                        if first:
                            observed["logits"] = out.logits[:, -1].clone()
                        first = False
                        ids = out.logits[:, -1].argmax(-1, keepdim=True)
                        generated.append(int(ids.item()))
                        cache = out.past_key_values
                trace.append({"stream": key, "event": "generation_complete"})
            except BaseException:
                for barrier in barriers:
                    barrier.abort()
                raise
            finally:
                for handle in handles:
                    handle.remove()
            observations[key] = observed
            records[key] = {
                "completion": tokenizers[key].decode(generated, clean_up_tokenization_spaces=False),
                "generated_tokens": generated,
                "finite": finite and all(bool(torch.isfinite(t).all()) for t in observed.values()),
                "rounds": rounds,
                "extra_expert_calls": extra_expert_calls,
                "seconds": round(time.time() - started, 2),
            }

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream, key) for key in SPECS]
            for future in futures:
                future.result()
        tensors[name] = observations
        runs[name] = {
            "active": active,
            "suppress_first_message": suppress_first,
            "streams": records,
            "trace": trace,
        }
        print(f"{name}: {json.dumps(records)}", flush=True)

    paired_run("baseline", False)
    paired_run("closed_loop", True)
    paired_run("suppress_first_message", True, True)
    paired_run("restored", False)
    full = tensors["closed_loop"]
    cut = tensors["suppress_first_message"]
    comparisons = {
        "A_first_response_max_abs": float(
            (full["A"]["round1_response"] - cut["A"]["round1_response"]).abs().max()
        ),
        "A_before_return_max_abs": float(
            (full["A"]["round2_boundary_before"] - cut["A"]["round2_boundary_before"]).abs().max()
        ),
        "first_delivered_message_delta_l2": float(
            (full["A"]["round1_sent"] - cut["A"]["round1_sent"]).norm()
        ),
        "suppressed_first_message_l2": float(cut["A"]["round1_sent"].norm()),
        "B_later_expert_input_delta_l2": float(
            (full["B"]["round2_expert_input"] - cut["B"]["round2_expert_input"]).norm()
        ),
        "B_return_response_delta_l2": float(
            (full["B"]["round2_response"] - cut["B"]["round2_response"]).norm()
        ),
        "A_received_return_delta_l2": float(
            (full["A"]["round2_received"] - cut["A"]["round2_received"]).norm()
        ),
        "A_projected_return_delta_l2": float(
            (full["A"]["round2_contribution"] - cut["A"]["round2_contribution"]).norm()
        ),
        "A_after_return_delta_l2": float(
            (full["A"]["round2_boundary_after"] - cut["A"]["round2_boundary_after"]).norm()
        ),
        "A_downstream_layer10_delta_l2": float(
            (full["A"]["downstream_layer10"] - cut["A"]["downstream_layer10"]).norm()
        ),
        "A_logits_delta_l2": float((full["A"]["logits"] - cut["A"]["logits"]).norm()),
        "B_downstream_layer10_delta_l2": float(
            (full["B"]["downstream_layer10"] - cut["B"]["downstream_layer10"]).norm()
        ),
        "B_logits_delta_l2": float((full["B"]["logits"] - cut["B"]["logits"]).norm()),
    }
    restoration = {}
    for key in SPECS:
        b, r = tensors["baseline"][key], tensors["restored"][key]
        restoration[key] = {
            "downstream_max_abs": float(
                (b["downstream_layer10"] - r["downstream_layer10"]).abs().max()
            ),
            "logits_max_abs": float((b["logits"] - r["logits"]).abs().max()),
            "tokens_equal": runs["baseline"]["streams"][key]["generated_tokens"]
            == runs["restored"]["streams"][key]["generated_tokens"],
        }
    ordering = {}
    for name in ("closed_loop", "suppress_first_message"):
        events = [(e["stream"], e["event"]) for e in runs[name]["trace"]]
        ordering[name] = (
            events.index(("A", "round1_message_ready"))
            < events.index(("B", "round1_message_injected"))
            < events.index(("B", "round2_message_ready"))
            < events.index(("A", "round2_message_injected"))
        )
    passed = (
        all(s["finite"] and s["rounds"] == 2 for r in runs.values() for s in r["streams"].values())
        and all(ordering.values())
        and comparisons["A_first_response_max_abs"] == 0
        and comparisons["A_before_return_max_abs"] == 0
        and comparisons["suppressed_first_message_l2"] == 0
        and all(
            comparisons[k] > 0
            for k in [
                "first_delivered_message_delta_l2",
                "B_later_expert_input_delta_l2",
                "B_return_response_delta_l2",
                "A_projected_return_delta_l2",
                "A_downstream_layer10_delta_l2",
                "A_logits_delta_l2",
            ]
        )
        and all(
            r["downstream_max_abs"] == 0 and r["logits_max_abs"] == 0 and r["tokens_equal"]
            for r in restoration.values()
        )
    )
    receipt = {
        "experiment": "One live A -> B -> A closed-loop exchange",
        "outcome": "PASS" if passed else "PARTIAL",
        "settings": settings.model_dump(mode="json"),
        "models": {
            k: {
                "model": v[0],
                "revision": v[1],
                "prompt": v[2],
                "prompt_ids": prompts[k].flatten().tolist(),
                "expert": "layer 7 expert 0 (same as reciprocal experiment)",
            }
            for k, v in SPECS.items()
        },
        "protocol": "Round 1: A layer-7 expert 0 on A layer-7 normalized MoE input -> B layer-7 residual. B runs native layers 8 and 9. Round 2: SAME B layer-7 expert 0 on B's live layer-9 normalized MoE input -> A layer-9 residual. A then runs native layer 10 onward. Both rounds use barriers within the same two live forward streams, before the first token is chosen.",
        "projections": "Same zero padding (B 1024 -> A 2048) and truncation (A 2048 -> B 1024) as reciprocity; alpha 0.01 by default. No learned adapters or norm matching.",
        "intervention": "Zero only the delivered first A -> B response; A's expert still computes its original response, B computes a fresh later return in both conditions. A has no other receiving edge before round 2.",
        "execution": "Both full backbones resident, two local threads, separate prompts/tokenizers/KV caches/generation histories. One ordered round trip during prefill and four native greedy generation steps per stream by default.",
        "ordering_verified": ordering,
        "comparisons": comparisons,
        "restoration": restoration,
        "runs": runs,
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "torch": torch.__version__,
            "transformers": __version__,
            "python": platform.python_version(),
            "seed": 0,
            "attention": "eager",
            "experts": "eager",
        },
        "training": "None; all pretrained parameters frozen; no saved learned adapters used",
        "limitations": "One causally ordered round trip, one pair of prompts, OLMoE/Granite only. No sustained-loop, stability, usefulness, semantic, or performance claim. Both expert operators use SwiGLU.",
    }
    (settings.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(tensors, settings.output / "activations.pt")
    print(
        json.dumps(
            {"outcome": receipt["outcome"], "comparisons": comparisons, "restoration": restoration},
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("Closed-loop causal path not fully observed; see receipt")


if __name__ == "__main__":
    main()
