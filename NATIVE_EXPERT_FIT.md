# Native-expert imitation — bounded result

One frozen DeepSeek expert was fit to the raw function of one frozen native
OLMoE expert through small learned input/output maps. Some held-out approximation
was learned. The parameter-matched bypass performed better, and the random and
scrambled donor paths were similar to the pretrained donor path.

**Scientific outcome: no observed pretrained-donor advantage in this run.** This
is a negative result for that comparison, not a claim that transfer is impossible.
No receiving-model expert was replaced, and no semantic capability was measured.

## The actual target

Target: `allenai/OLMoE-1B-7B-0924`, revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`, layer 7 expert 0.
Donor: `deepseek-ai/DeepSeek-V2-Lite`, revision
`604d5664dddd88a0433dbae533b7fe9472482de0`, layer 1 expert 0.
Both experts use SwiGLU, with input/output width 2048 and intermediate widths
1024 (native) and 1408 (donor).

The native forward implementation was intercepted at its expert boundary. Only
inputs actually routed to expert 0 were selected. The same native expert class
then evaluated those inputs with unit routing weight to record raw `E_0(x)`
before mixture weighting. The labels are neither whole-layer outputs nor final
receiving-model outputs. No residual injection participates in this experiment.

The fixed [Tiny Shakespeare corpus](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt)
was tokenized with OLMoE's tokenizer. Its first 8192 tokens yielded **399 training
pairs**; its final 4096 tokens yielded **262 held-out pairs**, separated by a large
unused region. Native contexts reset every 256 tokens. The split is by corpus
region before capture, rather than a random token split within adjacent text.
This does not guarantee lexical uniqueness or broad distribution coverage.

## Fixed training recipe

A and B are each a learned diagonal plus a rank-16 linear update, with no bias.
A starts as identity; B starts at zero. Only these maps train. The donor's
8,650,752 parameters remain frozen and receive no gradients. The bypass is
`2048 -> 33 -> SiLU -> 2048` with no biases and zero output initialization.
Every condition has exactly **135,168 trainable parameters**. Matching counts does
not make these different function classes equally expressive or equally easy to
optimize.

The secondary controls retain the same donor architecture: Gaussian random
weights with standard deviation 0.02 (seed 1), and independent scalar
permutations within each original matrix (seed 2). They are separate copies.

For numerical scaling, predictions and labels use the training-target RMS, and
each frozen expert uses a fixed output RMS measured on training x before
adaptation. In native units the foreign path is
`s_y * B(E(A(x)) / s_E)`. These frozen scalars use training pairs only; there is
no per-example norm matching or held-out calibration. They can be folded into
B's linear coefficients. The bypass predicts in the same target-RMS units.

Each condition received 300 Adam updates, learning rate 0.001, batch size 64,
initialization seed 3, and identical minibatch draws (seed 4). The final step was
selected in advance. Held-out measurements at steps 0, 100 and 300 did not affect
training, tuning or checkpoint selection. Capture ran on CPU; fitting ran on the
single available CUDA GPU, float32, deterministic algorithms, TF32 disabled.

## Measured approximation

MSE is averaged over native output coordinates and held-out pairs. Relative MSE
divides by the held-out target's mean squared magnitude (the zero-predictor
error). Cosine is the mean of per-pair output cosine similarities.

| Condition | Final train MSE | Final held-out MSE | Relative held-out MSE | Held-out cosine |
|---|---:|---:|---:|---:|
| Pretrained | 0.000811974 | 0.001119555 | 0.9105 | 0.2917 |
| Bypass | 0.000708662 | 0.001067691 | 0.8684 | 0.3452 |
| Random | 0.000808170 | 0.001112896 | 0.9051 | 0.2894 |
| Scrambled | 0.000808544 | 0.001121403 | 0.9121 | 0.2827 |

The zero predictor's held-out MSE is 0.001229537 (cosine 0).
The training-mean predictor's held-out MSE is
0.001182249, cosine
0.2026.
The bypass's final held-out MSE was 4.63% lower than the pretrained
foreign path's. Larger cosine also favored the bypass.

All conditions improved over their zero-output initialization. Foreign paths
continued reducing training error from step 100 to 300 while held-out MSE
slightly worsened. The fixed final checkpoint is reported regardless; no earlier
checkpoint was selected to improve the result. Pretrained, random and scrambled
conditions have similar errors in this single run; this is not a statistical
equivalence result.

## Artifacts and limits

Run `uv run python fit_native_expert.py`.

- [Executable](fit_native_expert.py)
- [Full receipt and learning measurements](results/native-expert-fit/receipt.json)
- [Recorded native input/output pairs](results/native-expert-fit/pairs.pt)
- [Learned adapter/bypass parameters and frozen scaling constants](results/native-expert-fit/adapters.pt)

The execution completed in 149.88 seconds. Ruff passes; no broad test
suite was added or run. Exact runtime versions and data ranges are in the receipt.

This is one native expert, one donor, one literary corpus split, one seed, and one
adapter/optimizer budget. It establishes a modest fit to a defined native expert
function, not successful substitution inside inference. It supplies no evidence
that pretrained donor structure makes this approximation easier or better than
the equally budgeted bypass. It does not test parameter efficiency or sample
efficiency across budgets. Both experts share SwiGLU and input/output widths.

Work stops here. A direct replacement run was not attempted. Any claim of a
consistent pretrained advantage would require further explicitly assigned
experiments; this run cannot support it.
