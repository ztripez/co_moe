# Consistent OLMoE/Granite replication

**R0 PASS · R1 PASS · R2 PASS.** R3 references the existing successful reciprocal
and feedback receipts; neither experiment was rerun or changed. All ten newly
executed OLMoE generations completed with finite recorded tensors and logits.
Historical DeepSeek receipts and implementations remain unchanged.

| Model | Pinned revision | Selected expert |
|---|---|---|
| `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | Layer 7, expert 0 (R2 target) |
| `ibm-granite/granite-3.1-1b-a400m-base` | `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29` | Layer 7, expert 0 (donor) |

These match the pinned reciprocal experiment. The selected pretrained Granite
weights were loaded directly from checkpoint slices and executed with Granite's
native expert implementation. No donor backbone was instantiated or executed.
Both expert operators use SwiGLU; Granite's intermediate width is 512.

## R0 — Foreign expert injection: PASS

Fixed maps truncate OLMoE's 2048-dimensional activation to 1024, then zero-pad
Granite's 1024-dimensional output to 2048. Add `0.01 * contribution` at OLMoE's
layer-7 residual output; observe layer 8 and final logits at the last prompt token.
The existing prompt `The capital of France is` and four greedy tokens are reused.

| Observation | L2 difference |
|---|---:|
| Active vs native, downstream activation | 0.00120284280 |
| Active vs native, logits | 1.07064581 |
| Negated donor input vs active, donor output | 0.00000944771182 |
| Negated donor input vs active, downstream activation | 0.00084401638 |
| Negated donor input vs active, logits | 0.73644137 |

Disabling the bridge restored downstream activations and logits with maximum
absolute error **0**, and identical generated token IDs.

## R1 — Donor weight controls: PASS

The same prompt, coefficient and fixed maps were used. Random weights use the
existing seed 1 and normal distribution (standard deviation 0.02). Scrambling
uses seed 2 with independent permutations within each original gate/up/down
matrix, preserving its empirical value distribution. Per-token projected
contribution norms match the pretrained response on that same activation,
before the coefficient. The already completed R0 active run supplies the
pretrained condition: its norm target is itself, giving scale exactly 1.

| Donor | Contribution L2 before coefficient | Downstream Δ L2 vs native | Logits Δ L2 vs native |
|---|---:|---:|---:|
| Pretrained | 0.000755439920 | 0.00120284280 | 1.07064581 |
| Random | 0.000755439978 | 0.00134457613 | 1.24703753 |
| Scrambled | 0.000755439862 | 0.00127603242 | 1.07288444 |

Maximum relative norm-matching error was **2.36e-7**, within the recorded 1e-5
numerical tolerance. Pre-injection activations were exactly equal. These are
observed computational differences, not a ranking by quality or evidence of a
pretraining advantage. No new prompt or seed was introduced.

## R2 — Live native-routed substitution: PASS

The exact 256-token context from the historical live-substitution receipt
(corpus range `[336144, 336400)`) was reused, followed by eight greedy tokens.
The native router selected expert 0 at the same six prompt positions. The Granite
path replaces only its **raw output before the original mixture weight**.
Input truncation/output padding are fixed; no adapter, norm matching or extra
0.01 coefficient is applied inside the substitution slot.

- Native routing indices and mixture weights remained exactly unchanged.
- Native branches for all other experts were retained. MoE rows not selecting
  expert 0 were exactly unchanged; the changed selected rows matched the raw
  replacement difference times the original weights within **3.58e-9** roundoff.
- Foreign vs native selected-position layer-8 Δ L2: **0.13577385**;
  last-prompt logits Δ L2: **0.44296810**.
- Foreign vs zero-expert last-prompt logits Δ L2: **0.09988840**.
- Restoring the original expert recovered recorded downstream activations and
  logits with maximum absolute error **0**, and identical generated tokens.

## R3 — Existing evidence: PASS, referenced only

- [Reciprocal inference receipt](results/reciprocal-inference/receipt.json)
- [Closed-loop feedback receipt](results/feedback-inference/receipt.json)

Their implementation was not modified or rerun, so the original evidence remains
applicable to these same checkpoint revisions.

## Differences from the DeepSeek experiments

The donor changes from DeepSeek layer 1 expert 0 (width 2048, intermediate 1408)
to Granite layer 7 expert 0 (width 1024, intermediate 512). Fixed truncation and
padding replace equal-width permutation maps. R2 uses fixed maps without the
learned imitation adapters used by the historical DeepSeek substitution run.
The demonstrated claim remains execution feasibility and causal influence,
without native functional equivalence or useful learned-transfer claims.

Run `uv run python replicate_granite_pair.py`.
[Full receipt](results/granite-pair-replication/receipt.json) ·
[Captured tensors](results/granite-pair-replication/activations.pt) ·
[Executable](replicate_granite_pair.py).
CPU float32, seed 0, 12 threads, eager attention/experts; Ruff passes. No training,
quality benchmark, optimization or broad test suite was performed. Replication
stops here.
