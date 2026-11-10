"""Fit frozen foreign computation to one native expert, with bounded controls."""

import argparse
import copy
import gc
import json
import os
import platform
import time
from pathlib import Path

import httpx
import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from co_moe0 import DONOR, DONOR_REV, PRIMARY, PRIMARY_REV, donor_expert

CORPUS_REV = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
CORPUS_URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{CORPUS_REV}/data/tinyshakespeare/input.txt"


class Settings(BaseModel):
    """Controls for this fixed, single-expert experiment."""

    output: Path = Path("results/native-expert-fit")
    steps: int = Field(default=300, ge=1)
    train_tokens: int = Field(default=8192, ge=256, multiple_of=256)
    heldout_tokens: int = Field(default=4096, ge=256, multiple_of=256)
    threads: int = Field(default=12, ge=1)


def capture_pairs(settings):
    """Record routed native inputs and evaluate their unweighted expert function."""
    corpus_path = settings.output / "corpus.txt"
    if not corpus_path.exists():
        response = httpx.get(CORPUS_URL, follow_redirects=True, timeout=60)
        response.raise_for_status()
        corpus_path.write_text(response.text)
    primary_path = snapshot_download(
        PRIMARY, revision=PRIMARY_REV, allow_patterns=["*.json", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(primary_path)
    tokens = tokenizer(corpus_path.read_text(), return_tensors="pt", verbose=False)["input_ids"]
    if tokens.shape[1] <= settings.train_tokens + settings.heldout_tokens:
        raise RuntimeError("Corpus too short for separated splits")
    splits = {
        "train": tokens[:, : settings.train_tokens],
        "heldout": tokens[:, -settings.heldout_tokens :],
    }
    model = (
        AutoModelForCausalLM.from_pretrained(
            primary_path, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .requires_grad_(False)
    )
    pairs = {}
    for split, ids in splits.items():
        xs, ys = [], []

        def capture(module, args, xs=xs, ys=ys):
            x, indices, _ = args
            x = x[(indices == 0).any(dim=-1)]
            if x.shape[0] == 0:
                return
            # Direct forward bypasses this hook; same native implementation,
            # selecting only expert 0 with unit routing weight gives raw E_0(x).
            y = module.forward(
                x, torch.zeros((len(x), 1), dtype=torch.long), torch.ones((len(x), 1))
            )
            xs.append(x.clone())
            ys.append(y.clone())

        handle = model.model.layers[7].mlp.experts.register_forward_pre_hook(capture)
        try:
            for offset in range(0, ids.shape[1], 256):
                with torch.inference_mode():
                    model.model(input_ids=ids[:, offset : offset + 256], use_cache=False)
                if (offset // 256 + 1) % 4 == 0:
                    print(
                        f"capture {split}: {offset + 256}/{ids.shape[1]} tokens; "
                        f"{sum(len(x) for x in xs)} routed pairs",
                        flush=True,
                    )
        finally:
            handle.remove()
        if not xs:
            raise RuntimeError(f"No tokens routed to selected native expert in {split}")
        pairs[f"{split}_x"] = torch.cat(xs)
        pairs[f"{split}_y"] = torch.cat(ys)
    metadata = {
        "corpus_url": CORPUS_URL,
        "corpus_token_count": tokens.shape[1],
        "train_token_range": [0, settings.train_tokens],
        "heldout_token_range": [tokens.shape[1] - settings.heldout_tokens, tokens.shape[1]],
        "context_tokens": 256,
        "context_reset_each_chunk": True,
        "native_expert": "model.layers.7.mlp.experts.0",
        "selection": "Only tokens actually routed to expert 0; raw expert outputs, before router weighting",
        "train_pairs": len(pairs["train_x"]),
        "heldout_pairs": len(pairs["heldout_x"]),
    }
    torch.save(pairs, settings.output / "pairs.pt")
    (settings.output / "data.json").write_text(json.dumps(metadata, indent=2) + "\n")
    del model
    gc.collect()
    return pairs, metadata


class SmallMap(nn.Module):
    """A learned diagonal plus rank-16 linear map in the original 2048-space."""

    def __init__(self, zero=False):
        super().__init__()
        self.diagonal = nn.Parameter(torch.zeros(2048) if zero else torch.ones(2048))
        self.down = nn.Linear(2048, 16, bias=False)
        self.up = nn.Linear(16, 2048, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """Apply the diagonal and low-rank branches."""
        return self.diagonal * x + self.up(self.down(x))


class ForeignFit(nn.Module):
    """Train only A and B around a frozen expert with fixed train-set scaling."""

    def __init__(self, expert, expert_rms):
        super().__init__()
        self.expert = expert
        self.input_map = SmallMap()
        self.output_map = SmallMap(zero=True)
        self.register_buffer("expert_rms", expert_rms)

    def forward(self, x):
        """Predict in units of the training target RMS."""
        return self.output_map(self.expert(self.input_map(x)) / self.expert_rms)


@torch.no_grad()
def evaluate(model, x, y, target_rms):
    """Measure raw output MSE, normalized MSE, and mean per-pair cosine."""
    predictions = torch.cat([model(batch) * target_rms for batch in x.split(128)])
    mse = F.mse_loss(predictions, y)
    return {
        "mse": float(mse),
        "relative_mse": float(mse / y.square().mean()),
        "mean_cosine": float(F.cosine_similarity(predictions, y, dim=-1).mean()),
    }


def main():
    """Capture one function, fit four conditions, and save their fixed-budget results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/native-expert-fit"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-tokens", type=int, default=8192)
    parser.add_argument("--heldout-tokens", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=12)
    settings = Settings.model_validate(vars(parser.parse_args()))
    settings.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    pairs, data_metadata = capture_pairs(settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = {k: v.to(device) for k, v in pairs.items()}
    x, y = pairs["train_x"], pairs["train_y"]
    hx, hy = pairs["heldout_x"], pairs["heldout_y"]
    target_rms = y.square().mean().sqrt()
    pretrained, _, config = donor_expert()
    random = copy.deepcopy(pretrained)
    scrambled = copy.deepcopy(pretrained)
    rg = torch.Generator().manual_seed(1)
    sg = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for parameter in random.parameters():
            parameter.normal_(0, config["initializer_range"], generator=rg)
        for parameter in scrambled.parameters():
            parameter.copy_(
                parameter.flatten()[torch.randperm(parameter.numel(), generator=sg)].reshape_as(
                    parameter
                )
            )
    results, checkpoints = {}, {}
    for name, expert in [
        ("pretrained", pretrained),
        ("bypass", None),
        ("random", random),
        ("scrambled", scrambled),
    ]:
        torch.manual_seed(3)
        if expert is None:
            model = nn.Sequential(
                nn.Linear(2048, 33, bias=False), nn.SiLU(), nn.Linear(33, 2048, bias=False)
            ).to(device)
            nn.init.zeros_(model[-1].weight)
            expert_rms = None
        else:
            expert = expert.to(device).eval().requires_grad_(False)
            with torch.no_grad():
                expert_rms = (
                    torch.cat([expert(batch) for batch in x.split(128)]).square().mean().sqrt()
                )
            model = ForeignFit(expert, expert_rms).to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        count = sum(p.numel() for p in parameters)
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        batches = torch.Generator().manual_seed(4)
        history = []
        for step in range(settings.steps + 1):
            if step in {0, 100, settings.steps}:
                row = {
                    "step": step,
                    "train": evaluate(model, x, y, target_rms),
                    "heldout": evaluate(model, hx, hy, target_rms),
                }
                history.append(row)
                print(f"{name}: {json.dumps(row)}", flush=True)
            if step == settings.steps:
                break
            ids = torch.randint(len(x), (64,), generator=batches).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[ids]), y[ids] / target_rms)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss for {name} at step {step}")
            loss.backward()
            optimizer.step()
        results[name] = {
            "trainable_parameters": count,
            "history": history,
            "expert_rms": None if expert_rms is None else float(expert_rms),
            "frozen_expert_parameters": 0
            if expert is None
            else sum(p.numel() for p in expert.parameters()),
            "expert_has_gradients": False
            if expert is None
            else any(p.grad is not None for p in expert.parameters()),
        }
        checkpoints[name] = {
            "parameters": {
                n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
            },
            "target_rms": target_rms.cpu(),
            "expert_rms": None if expert_rms is None else expert_rms.cpu(),
        }
        del model, optimizer
    with torch.no_grad():
        mean_prediction = y.mean(dim=0, keepdim=True).expand_as(hy)
        mean_baseline = {
            "mse": float(F.mse_loss(mean_prediction, hy)),
            "mean_cosine": float(F.cosine_similarity(mean_prediction, hy).mean()),
        }
    receipt = {
        "experiment": "Frozen DeepSeek expert imitation of one frozen OLMoE expert",
        "outcome": "COMPLETED",
        "settings": settings.model_dump(mode="json"),
        "primary": {
            "model": PRIMARY,
            "revision": PRIMARY_REV,
            "expert": data_metadata["native_expert"],
        },
        "donor": {"model": DONOR, "revision": DONOR_REV, "expert": "model.layers.1.mlp.experts.0"},
        "data": data_metadata,
        "results": results,
        "train_mean_heldout_baseline": mean_baseline,
        "heldout_zero_mse": float(hy.square().mean()),
        "target_train_rms": float(target_rms),
        "maps": "A/B: learned diagonal plus rank-16 linear update, no bias; A initialized identity, B zero. Bypass: 2048 -> 33 -> SiLU -> 2048, no bias, zero output initialization.",
        "scaling": "Raw x. Loss in train-target-RMS units; each frozen expert output divided by its fixed RMS on training x at identity A. Prediction in native units = target_rms * B(E(A(x))/expert_rms). Scalars are frozen and use train data only.",
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "steps": settings.steps,
            "initialization_seed": 3,
            "batch_seed": 4,
            "random_expert_seed": 1,
            "scramble_seed": 2,
        },
        "checkpoint_selection": "Final fixed-budget step; heldout metrics never used for selection or training",
        "runtime": {
            "device": str(device),
            "dtype": "float32",
            "torch": torch.__version__,
            "transformers": __version__,
            "python": platform.python_version(),
        },
        "seconds": round(time.time() - started, 2),
        "limitations": "One native expert, one donor, one corpus split, one training seed and adapter budget; no sample-efficiency or pretraining-transfer claim; no in-model replacement run.",
    }
    (settings.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(checkpoints, settings.output / "adapters.pt")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
