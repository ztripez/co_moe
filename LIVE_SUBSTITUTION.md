# Live native-expert substitution — PASS

A frozen pretrained DeepSeek expert, using the existing saved input/output maps,
occupied OLMoE layer 7 expert 0's execution slot during real inference. Native
routing selected the slot, the receiving model consumed the replacement under
the original mixture weights, and inference completed. This demonstrates
functional substitution as execution feasibility, without requiring equivalence
or improved output quality.

Run: `uv run python substitute_expert.py`

## Execution

Primary: `allenai/OLMoE-1B-7B-0924`, checkpoint
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
Donor: `deepseek-ai/DeepSeek-V2-Lite`, checkpoint
`604d5664dddd88a0433dbae533b7fe9472482de0`, layer 1 expert 0.
Maps and bypass: the previously saved step-300 parameters from
`results/native-expert-fit/adapters.pt`. This was the available persisted
checkpoint; no earlier checkpoint was selected and no further training occurred.
The training-time fixed scaling constants are restored unchanged.

The first 256-token context from the previously held-out corpus region
(tokens 336144–336399, end-exclusive range `[336144, 336400)`) already exercised
the target. Context selection depended only on native routing. Expert 0 was
selected at prompt positions **58, 70, 80, 147, 148 and 237**. Its original mixture
weights ranged from **0.0245992 to 0.0313615**.

The replacement branch changes only expert 0's raw output, before mixture
weighting. All other expert branches execute native computations. The router
is untouched. No additive augmentation, coefficient tuning, or contribution
norm matching is used. Both models' pretrained weights and the saved bridge
parameters remain frozen. The donor backbone never executes.

Each replacement was executed once as a batch of six routed prompt activations.
There were no further expert-0 selections during the eight-token continuation;
substitution in prefill is part of the live inference and affects subsequent
computation and cached state.

## Observed evidence

All five conditions completed with finite recorded expert/MoE/downstream outputs
and logits. On the identical prompt, router indices and mixture weights matched
native execution exactly. MoE rows not selecting expert 0 were also unchanged
exactly. The measured mixture-output differences matched the raw replacement
change multiplied by the original weights, within 5.48e-9 absolute roundoff.

| Condition | Layer 8 Δ L2 at selected prompt positions | Last-prompt logits Δ L2 |
|---|---:|---:|
| Native | 0 | 0 |
| Frozen foreign path | 0.12237718 | 0.48000166 |
| Saved trained bypass | 0.12014207 | 0.45832291 |
| Zero expert | 0.13044317 | 0.45292050 |
| Restored native | 0 | 0 |

Foreign versus zero last-prompt logits differed by L2 **0.28165132**. The
foreign raw output was nonzero (L2 **1.59484756** across the six selected tokens),
so the foreign condition did more than remove the original contribution.

Restoring the original expert recovered the full recorded layer 8 activation
and last-prompt logits with maximum absolute error **0**, and identical generated
tokens. Every condition generated `\nUpon this island as a spy,`. Matching tokens
is incidental, not an acceptance requirement or quality finding.

## Artifacts and limits

- [Executable](substitute_expert.py)
- [Machine-readable receipt](results/live-substitution/receipt.json)
- [Recorded routing, weights, raw outputs and downstream tensors](results/live-substitution/activations.pt)
- [Research charter](CHARTER.md)

Execution used one machine, CPU float32, seed 0, deterministic algorithms,
12 threads, eager attention and eager expert execution. Ruff passes. No broad
suite, training campaign, benchmark, or quality optimization was run.

An initial run completed substitution and exact restoration, but its receipt
comparison mixed expert kernels and raw-output row orders. It is retained under
`results/live-substitution-initial/` with an explanatory note. The final run uses
eager experts consistently and aligns the recorded rows; the bridge is unchanged.

This establishes execution-slot substitution for the tested model pair, expert,
context and bridge. It does not establish native functional equivalence, useful
pretrained transfer, or quality improvement. Both experts still use SwiGLU and
matching input/output widths. Bidirectional independent-model exchange remains
unimplemented and is a separate future assignment. Work stops at substitution.
