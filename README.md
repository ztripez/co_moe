# Co-MoE-0

One CPU inference path through OLMoE, augmented by one frozen, pretrained
DeepSeek-V2-Lite expert. No donor backbone is instantiated or executed.

```sh
uv run python co_moe0.py
```

The script runs baseline, coupled, coupled repeat, negated donor input, and
restored baseline. Defaults: one five-token prompt, four greedy continuation
tokens, alpha 0.01, seed 0, 12 CPU threads, float32. It downloads about 14 GB of
primary weights into the Hugging Face cache and only about 17 MB of donor expert
weights using HTTP byte ranges. Allow roughly 30 GiB RAM for float32 inference.
It leaves existing model stores untouched. `--output` selects the receipt folder;
`--alpha`, `--prompt`, `--tokens`, and `--threads` adjust this fixed experiment.

The exact checkpoint revisions are in the script and receipts. The donor loader
executes only the publisher's standalone `DeepseekV2MLP` class extracted from its
source, then loads the three pretrained expert matrices. It never constructs the
donor attention, router, tokenizer, or model. All model parameters are frozen;
there is no training or source-weight write.

The primary layer 7 output hook adds `alpha * P_out(E(P_in(h)))` to its completed
native residual update. Layer 8 continues normally. Both projections are fixed
2048-dimensional permutations: cyclic shift +17 and its inverse. The donor input
intervention negates the projected activation while keeping alpha and weights
fixed. Causal comparisons use the same final prompt position, before generated
token histories can differ. Disabling the bridge skips donor execution entirely.

See [RESULT.md](RESULT.md), the JSON receipts, and saved activation tensors under
`results/`. The program reports PASS only when the finite completed runs show a
nonzero donor-dependent downstream change, an exact active repeat, and exact
baseline restoration (including logits and generated tokens).

The scope ends at computational coupling. Semantic usefulness, reciprocal
cooperation, expert substitution, and adapter training have not been tested.

## Donor weight controls

`uv run python co_moe0.py --weight-controls` runs native, pretrained, random, and
scrambled conditions with per-token contribution norm matching. Results go to
`results/weight-controls/` by default. See [WEIGHT_CONTROLS.md](WEIGHT_CONTROLS.md)
for the observed differences and their limits. No adapters are trained.

## Fit one native expert

`uv run python fit_native_expert.py` records inputs actually routed to OLMoE
layer 7 expert 0, evaluates that expert's raw output, and fits input/output maps
around the frozen DeepSeek expert. A parameter-matched bypass and frozen
random/scrambled donor controls use the same data and training budget. Held-out
pairs come from a separate corpus region; this is expert imitation, without
additive residual injection or in-model expert replacement.

Each condition has 135,168 trainable parameters and runs 300 Adam updates.
Both pretrained experts stay frozen. Native capture uses CPU; fitting uses CUDA
when available. All losses and final metrics are saved under
`results/native-expert-fit/`, alongside recorded pairs and adapter parameters.
See [NATIVE_EXPERT_FIT.md](NATIVE_EXPERT_FIT.md) for results and limitations.

## Live native-expert substitution

`uv run python substitute_expert.py` loads the saved foreign adapters and bypass
without training. It selects a held-out context solely because the native router
selects layer 7 expert 0, then runs native, foreign, bypass, zero, and restored
conditions. Only that expert's raw output is replaced; native routing and its
original mixture weights remain in use. The donor backbone never executes.

See [LIVE_SUBSTITUTION.md](LIVE_SUBSTITUTION.md) for the execution receipt and
[CHARTER.md](CHARTER.md) for the distinction between feasibility and optimization.

## Reciprocal live inference

`uv run python reciprocal_inference.py` runs full OLMoE and Granite MoE backbones
with separate prompts, tokenizers, caches and generation histories. At one
shared prefill pause point, each sends a live expert response into the other
backbone. Two directional suppression interventions demonstrate reciprocal
causal effects; disabling exchange checks native restoration. No training or
learned adapters are used. This changes the second full model from DeepSeek to
Granite to fit local memory. See [RECIPROCAL_INFERENCE.md](RECIPROCAL_INFERENCE.md).

## One closed-loop exchange

`uv run python feedback_inference.py` performs one ordered A→B→A round trip using
the existing OLMoE/Granite pair and the same two auxiliary experts. It suppresses
the first delivered message to trace its effect through Granite's later
computation and back into OLMoE, then restores both native baselines. There is no
training or quality optimization. See [FEEDBACK_INFERENCE.md](FEEDBACK_INFERENCE.md).

## Consistent model-pair replication

`uv run python replicate_granite_pair.py` repeats injection, norm-matched donor
weight controls and native-routed substitution with the pinned OLMoE/Granite pair,
fixed truncation/padding maps and no learned adapters. Historical DeepSeek
receipts remain unchanged; successful reciprocal and feedback runs are referenced
without rerunning. See [GRANITE_PAIR_REPLICATION.md](GRANITE_PAIR_REPLICATION.md).
