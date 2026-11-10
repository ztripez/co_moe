# Co-MoE-0 — PASS

A genuine pretrained DeepSeek-V2-Lite expert processed an OLMoE residual
activation, and its projected output causally changed OLMoE's subsequent
computation. All five short inference runs completed with finite recorded tensors
and logits. Disabling the bridge restored the native result exactly.

| Field | Observation |
|---|---|
| Primary | `allenai/OLMoE-1B-7B-0924`, revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| Donor | `deepseek-ai/DeepSeek-V2-Lite`, revision `604d5664dddd88a0433dbae533b7fe9472482de0` |
| Expert | `model.layers.1.mlp.experts.0`, original gate/up/down matrices |
| Injection | Zero-based primary layer 7 output, after native attention and MoE residual additions |
| Observation | Layer 8 output and final logits, at the last prompt token |
| Shapes | Prefill residual/input/output/contribution `[1, 5, 2048]`; decode `[1, 1, 2048]`; donor intermediate width 1408 |
| Projection | Fixed cyclic permutation +17 on input, inverse permutation on output; 2048 → 2048 |
| Coefficient | `alpha = 0.01` |
| Prompt | `The capital of France is` |
| Completion | Baseline, coupled, intervention, repeat, restored: ` Paris.\n\n` |
| Intervention | Negate donor input; retain pretrained weights, projections, coefficient, and primary input |
| Restoration | Downstream activation and logits exactly equal baseline; token IDs identical |
| Donor backbone | Never instantiated or executed; only standalone publisher MLP class and three expert matrices loaded |
| Outcome | PASS: computational coupling, with no semantic usefulness claim |

The models are independently pretrained architectures, not fine-tunes of one
base. [OLMoE's configuration](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/6d84c48581ece794365f2b8e9cfb043c68ade9c5/config.json)
has 16 layers, standard attention, 64 routed experts, top-8 routing, and expert
intermediate width 1024.
[DeepSeek-V2-Lite's configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json)
has 27 layers, latent KV compression, top-6 routing over 64 routed experts plus
two shared experts, and routed-expert width 1408. Both selected expert designs
use SwiGLU; this experiment does **not** establish interoperability between
different activation functions or gating formulas. Equal residual widths allow
simple fixed permutation projections without padding or truncation.

The loader selects only `DeepseekV2MLP` from the
[publisher's implementation](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py).
Its native computation is `down(silu(gate(x)) * up(x))`. Only that expert's
checkpoint byte ranges are fetched. No donor routing or backbone is needed.

At the fixed prompt position, coupled versus baseline downstream L2 difference
was **0.00986246765** and logit L2 difference was **8.51207447**. Negating the donor
input changed the actual donor output by L2 **0.00042793524**, and changed the
receiving downstream activation by **0.01168186869** and logits by
**10.55937862**, relative to the ordinary coupled run. Thus the measured effect
depends on donor computation rather than a fixed injected constant. The active
repeat and disabled restoration both had maximum absolute downstream difference
**0**. Restored logits also had maximum absolute difference **0**.

Rerun with `uv run python co_moe0.py`. The standalone uv environment was executed
successfully with Python 3.13.7, Torch 2.15.0, Transformers 5.22.2, CPU float32,
12 threads, seed 0, deterministic algorithms, and exact restoration tolerance.
Its receipt and raw captured tensors are in
[results/co-moe-0-uv/](results/co-moe-0-uv/). The initial execution using existing
Torch 2.11.0 / Transformers 5.7.0 is retained under
[results/co-moe-0/](results/co-moe-0/); it produced the same reported differences.
Ruff passes. No broad test suite was added or run.

This establishes the specified mechanical coupling for one expert, boundary,
prompt, and coefficient. Unchanged greedy text does not imply unchanged
computation; neither the changed logits nor successful coupling demonstrate
useful cooperation. No training or performance comparison was performed.

The smallest next experiment would replace both permutation maps with identity
maps on this same pair, to observe whether that single bridge choice changes the
measured contribution. It has not been run. Work stops at Co-MoE-0.
