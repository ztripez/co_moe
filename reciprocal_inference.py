"""One synchronized exchange between two resident, independently generating MoE models."""

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
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from co_moe0 import PRIMARY, PRIMARY_REV

GRANITE = "ibm-granite/granite-3.1-1b-a400m-base"
GRANITE_REV = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
SPECS = {
    "A": (PRIMARY, PRIMARY_REV, "The capital of France is"),
    "B": (GRANITE, GRANITE_REV, "The largest planet is"),
}


class Settings(BaseModel):
    """Controls for one exchange point per paired inference run."""

    output: Path = Path("results/reciprocal-inference")
    tokens: int = Field(default=4, ge=1)
    alpha: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    threads: int = Field(default=6, ge=1)


def project(vector, width):
    """Pad with zeros or truncate the expert response to the recipient width."""
    return F.pad(vector, (0, max(0, width - vector.shape[-1])))[..., :width]


def main():
    """Run a native pair, coupled pair, two directional interventions, and restoration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reciprocal-inference"))
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
        print(
            f"Loaded {key}: {sum(p.numel() for p in models[key].parameters())} parameters",
            flush=True,
        )
    moes = {"A": models["A"].model.layers[7].mlp, "B": models["B"].model.layers[7].block_sparse_moe}
    runs, tensors = {}, {}

    def paired_run(name, active, suppress=None):
        barrier = threading.Barrier(2)
        messages, observations, records, trace = {}, {}, {}, []

        def stream(key):
            other = "B" if key == "A" else "A"
            model = models[key]
            layer = model.model.layers[7]
            observed = {}
            first = True
            finite = True
            expert_calls = 0
            exchange_count = 0

            def expert_input(module, args):
                if first:
                    observed["expert_input"] = args[0][:, -1].clone()

            def exchange(module, args, output):
                nonlocal finite, expert_calls, exchange_count
                if not first:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                observed["boundary_before"] = h[:, -1].clone()
                trace.append({"stream": key, "event": "prefix_ready"})
                if active:
                    x = observed["expert_input"]
                    # Execute the publisher's native expert implementation on
                    # this live stream's own normalized MoE input, with unit weight.
                    response = moes[key].experts.forward(
                        x, torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1))
                    )
                    expert_calls += 1
                    observed["expert_response"] = response.clone()
                    messages[key] = (
                        torch.zeros_like(response) if suppress == key else response.clone()
                    )
                    observed["sent"] = messages[key].clone()
                    trace.append({"stream": key, "event": "message_ready"})
                barrier.wait(timeout=120)
                exchange_count += 1
                trace.append({"stream": key, "event": "resume"})
                if not active:
                    return output
                received = messages[other]
                contribution = project(received, h.shape[-1])
                changed = h.clone()
                changed[:, -1] += settings.alpha * contribution
                observed["received"] = received.clone()
                observed["contribution"] = contribution.clone()
                observed["boundary_after"] = changed[:, -1].clone()
                finite = finite and bool(torch.isfinite(changed).all())
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            def downstream(module, args, output):
                nonlocal finite
                h = output[0] if isinstance(output, tuple) else output
                finite = finite and bool(torch.isfinite(h).all())
                if first:
                    observed["downstream"] = h[:, -1].clone()

            handles = [
                moes[key].register_forward_pre_hook(expert_input),
                layer.register_forward_hook(exchange),
                model.model.layers[8].register_forward_hook(downstream),
            ]
            ids, cache, generated = prompts[key], None, []
            started = time.time()
            try:
                with torch.inference_mode():
                    for _ in range(settings.tokens):
                        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
                        finite = finite and bool(torch.isfinite(out.logits).all())
                        if first:
                            observed["logits"] = out.logits[:, -1].clone()
                        first = False
                        ids = out.logits[:, -1].argmax(-1, keepdim=True)
                        generated.append(int(ids.item()))
                        cache = out.past_key_values
                trace.append({"stream": key, "event": "generation_complete"})
            except BaseException:
                barrier.abort()
                raise
            finally:
                for handle in handles:
                    handle.remove()
            observations[key] = observed
            finite = finite and all(bool(torch.isfinite(t).all()) for t in observed.values())
            records[key] = {
                "completion": tokenizers[key].decode(generated, clean_up_tokenization_spaces=False),
                "generated_tokens": generated,
                "finite": finite,
                "extra_expert_calls": expert_calls,
                "exchange_points": exchange_count,
                "seconds": round(time.time() - started, 2),
            }

        # Two independent model objects, local thread stacks, prompts, tokenizers,
        # caches and generation histories remain alive across the same barrier.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream, key) for key in SPECS]
            for future in futures:
                future.result()
        tensors[name] = observations
        runs[name] = {
            "active": active,
            "suppressed_outbound_stream": suppress,
            "streams": records,
            "trace": trace,
        }
        print(f"{name}: {json.dumps(records)}", flush=True)

    paired_run("baseline", False)
    paired_run("coupled", True)
    paired_run("suppress_A_to_B", True, "A")
    paired_run("suppress_B_to_A", True, "B")
    paired_run("restored", False)
    comparisons = {}
    for key in SPECS:
        baseline = tensors["baseline"][key]
        coupled = tensors["coupled"][key]
        restored = tensors["restored"][key]
        intervention_name = "suppress_B_to_A" if key == "A" else "suppress_A_to_B"
        intervention = tensors[intervention_name][key]
        other_intervention_name = "suppress_A_to_B" if key == "A" else "suppress_B_to_A"
        other_intervention = tensors[other_intervention_name][key]
        comparisons[key] = {
            "coupled_downstream_delta_l2": float(
                (coupled["downstream"] - baseline["downstream"]).norm()
            ),
            "coupled_logits_delta_l2": float((coupled["logits"] - baseline["logits"]).norm()),
            "incoming_intervention_downstream_delta_l2": float(
                (intervention["downstream"] - coupled["downstream"]).norm()
            ),
            "incoming_intervention_logits_delta_l2": float(
                (intervention["logits"] - coupled["logits"]).norm()
            ),
            "incoming_suppressed_contribution_l2": float(intervention["contribution"].norm()),
            "outgoing_intervention_own_logits_max_abs": float(
                (other_intervention["logits"] - coupled["logits"]).abs().max()
            ),
            "outgoing_intervention_own_downstream_max_abs": float(
                (other_intervention["downstream"] - coupled["downstream"]).abs().max()
            ),
            "native_prefix_input_max_abs": float(
                (coupled["expert_input"] - baseline["expert_input"]).abs().max()
            ),
            "restored_downstream_max_abs": float(
                (restored["downstream"] - baseline["downstream"]).abs().max()
            ),
            "restored_logits_max_abs": float((restored["logits"] - baseline["logits"]).abs().max()),
            "restored_tokens_equal": runs["restored"]["streams"][key]["generated_tokens"]
            == runs["baseline"]["streams"][key]["generated_tokens"],
            "sent_shape": list(coupled["sent"].shape),
            "received_shape": list(coupled["received"].shape),
            "sent_l2": float(coupled["sent"].norm()),
            "contribution_l2": float(coupled["contribution"].norm()),
        }
    passed = all(
        s["finite"] and s["exchange_points"] == 1
        for r in runs.values()
        for s in r["streams"].values()
    ) and all(
        c["coupled_downstream_delta_l2"] > 0
        and c["incoming_intervention_downstream_delta_l2"] > 0
        and c["incoming_intervention_logits_delta_l2"] > 0
        and c["incoming_suppressed_contribution_l2"] == 0
        and c["restored_downstream_max_abs"] == 0
        and c["restored_logits_max_abs"] == 0
        and c["restored_tokens_equal"]
        for c in comparisons.values()
    )
    receipt = {
        "experiment": "Two live MoE streams with reciprocal expert-response exchange",
        "outcome": "PASS" if passed else "PARTIAL",
        "settings": settings.model_dump(mode="json"),
        "models": {
            k: {
                "model": spec[0],
                "revision": spec[1],
                "prompt": spec[2],
                "prompt_ids": prompts[k].flatten().tolist(),
                "hidden_size": models[k].config.hidden_size,
                "expert": "model.layers.7.mlp.experts.0"
                if k == "A"
                else "model.layers.7.block_sparse_moe.experts.0",
            }
            for k, spec in SPECS.items()
        },
        "execution": "Both full backbones resident and independently generating in two local threads, with one shared prefill barrier. Separate tokenizers, prompts, caches, generation histories. No network or subprocess transport.",
        "exchange": "At layer 7 output, each stream sends E_own(x_own) from its actual normalized layer-7 MoE input at its last prompt position. Recipient pads/truncates that expert response and adds alpha times the response to its own layer-7 residual at its own last prompt position. Both prefixes and messages exist before either coupled suffix resumes.",
        "interventions": "Suppress A's outbound expert response or B's outbound expert response, retaining the opposite direction in the same paired run. No response recomputation from recorded offline activations.",
        "training": "None; no saved learned adapters used",
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "torch": torch.__version__,
            "transformers": __version__,
            "python": platform.python_version(),
            "seed": 0,
            "attention_implementation": "eager",
            "experts_implementation": "eager",
        },
        "runs": runs,
        "comparisons": comparisons,
        "limitations": "OLMoE/Granite pair, not a live OLMoE/DeepSeek pair. One synchronous exchange per direction; no repeated feedback loop, semantic alignment, usefulness, or performance claim. Both expert operators use SwiGLU.",
    }
    (settings.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(tensors, settings.output / "activations.pt")
    print(
        json.dumps({"outcome": receipt["outcome"], "comparisons": comparisons}, indent=2),
        flush=True,
    )
    if not passed:
        raise RuntimeError("Reciprocal causal criteria not all observed; see receipt")


if __name__ == "__main__":
    main()
