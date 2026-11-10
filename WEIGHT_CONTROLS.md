# Donor weight controls — completed

The pretrained expert, an equally shaped random expert, and an independently
scrambled copy of the pretrained expert all affected OLMoE computation at matched
contribution norms. Their effects were distinguishable on the tested prompt.
These observations do not establish an advantage from pretraining.

Run:

```sh
uv run python co_moe0.py --weight-controls
```

The model checkpoints, frozen primary model, expert class, prompt, layer 7
injection, layer 8 observation, permutations, alpha 0.01, four greedy tokens,
CPU float32 execution, and deterministic settings match Co-MoE-0. Original
Co-MoE-0 receipts are retained. The new receipt is
[results/weight-controls/receipt.json](results/weight-controls/receipt.json), with
raw last-prompt-position tensors in `activations.pt` beside it.

A is native inference. B uses the genuine pretrained donor expert. C uses the
same expert class with independent Gaussian weights (mean 0, standard deviation
0.02 from the donor configuration's initializer range; seed 1). D independently
permutes every scalar entry within each original weight matrix (seed 2),
preserving each matrix's exact empirical value distribution while disrupting
its connectivity. All changes apply to separate in-memory copies. The donor
backbone remains inactive and pretrained checkpoint files remain untouched.

For each token, B/C/D outputs are scaled to the norm of the pretrained expert's
output on that same receiving activation, after output projection and before
alpha. B's scale is consequently 1. This normalization removes contribution
magnitude as the main confound; it is an experimental control, not learned
alignment. The maximum relative norm-matching error across prompt and decode
was 1.90e-7. All conditions had exactly identical pre-injection prompt activations.

Measurements below are at the same final prompt token, before generated token
histories can differ. Baseline downstream L2 norm was 1.094080; baseline logit L2
norm was 1081.760864. Relative logit changes provide a reference scale, not a
semantic significance measure; logit L2 is also sensitive to common offsets.

| Condition | Contribution L2 before alpha | Downstream Δ L2 vs native | Logits Δ L2 vs native | Relative logits Δ | Cosine of logit change with B |
|---|---:|---:|---:|---:|---:|
| A Native | 0 | 0 | 0 | 0% | undefined |
| B Pretrained | 0.011496406 | 0.009862468 | 8.512074 | 0.7869% | 1 |
| C Random | 0.011496406 | 0.009708921 | 8.844928 | 0.8176% | 0.0541 |
| D Scrambled | 0.011496406 | 0.010485767 | 12.330960 | 1.1399% | -0.2863 |

The actual added signal's L2 was approximately 0.000114964 in all three active
conditions. Before subsequent primary-model computation, random/pretrained
contribution cosine was -0.00222 and scrambled/pretrained cosine was -0.00612.
Thus downstream distinguishability accompanied different contribution directions
at equal magnitude. A larger logit change is not better performance.

All four runs completed with finite recorded tensors and logits. Every condition
produced token IDs `[7785, 15, 187, 187]`, decoded as ` Paris.\n\n`. The pretrained
condition reproduced the previously reported downstream and logit differences.
Ruff passes; no broad suite or training was run.

This is one prompt, one expert, and one random/scramble seed each. Distinct
outputs from different weights are expected; this is not statistical evidence
of a special or useful pretrained computation. The result confirms why the
original coupling claim cannot imply that pretraining is necessary. Both donor
and native experts still use SwiGLU with matching input/output widths.

The next proposed functional experiment is frozen-expert imitation: fit only
small input/output maps to recorded native expert input/output pairs, then
measure approximation on held-out pairs. An equally sized trainable bypass
control would distinguish what the foreign expert contributes from what the
adapters can learn themselves. Adapter training has not started.
