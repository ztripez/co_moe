# Co-MoE

Pretrained experts from different language models executing inside the same
inference graph. Co-MoE tests whether a frozen expert from one independently
trained mixture-of-experts (MoE) model can participate in another model's
computation, using fixed dimensional maps and unchanged native routing.

The consistent demonstration pair is **OLMoE / IBM Granite**. The experiments
establish execution and numerical causal influence under the tested conditions.
They do not establish useful knowledge transfer or improved model capability.

**[Read the technical report](output/pdf/co-moe-technical-report.pdf)** ·
[Manuscript](paper/manuscript.md) · [Research charter](CHARTER.md)

## What was demonstrated

- **Injection:** a standalone Granite expert processes an OLMoE activation and
  contributes to OLMoE's residual stream.
- **Routed substitution:** a Granite response replaces a selected native
  expert's raw output while retaining OLMoE's routing indices, mixture weights,
  and other expert contributions.
- **Reciprocity:** two live model backbones exchange expert responses produced
  from their own ongoing activations.
- **Feedback:** one ordered A → B → A exchange lets a first-message intervention
  change the returning response and the originating model's later computation.

These feasibility runs complete generation with finite monitored tensors.
Interventions change downstream computation; disabling the mechanisms restores
the recorded native baselines exactly in the tested runtime.

A denser test replaces four of eight routed contributions at one layer with
weighted copies of **one** Granite expert response. It completes across three
fixed masks, but prediction quality worsens on the evaluated text slice.
Zeroing the identical contributions is less damaging. See the
[50/50 results](HALF_SPLIT_BENCHMARK.md) and
[matched-zero control](HALF_SPLIT_ZERO.md).

Both expert implementations use SwiGLU. Different model families and hidden
widths do not establish compatibility between arbitrary expert operators. One
return path does not establish sustained feedback or stability.

## Models

| Role | Checkpoint | Pinned revision |
| --- | --- | --- |
| Primary | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| Donor / second live model | `ibm-granite/granite-3.1-1b-a400m-base` | `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29` |

The fixed bridge truncates OLMoE's 2048-dimensional input to 1024 and zero-pads
Granite's output back to 2048. The selected donor is Granite layer 7, expert 0;
the native substitution target is OLMoE layer 7, expert 0. Indices are zero-based.
Injection uses a coefficient of 0.01; substitution keeps the slot's original
mixture weight without an additional coefficient. No adapters are trained for
the OLMoE/Granite feasibility experiments.

## Run the consistent-pair experiment

Use Python 3.12 or newer and `uv`. These scripts run on one machine in CPU
float32. Allow roughly 30 GiB RAM for the OLMoE inference path; reciprocal and
feedback runs also keep Granite's full backbone resident and were executed on a
64 GiB machine. Checkpoint downloads require several GB of disk space and are
cached by Hugging Face. Pretrained model weights are not included in this repo.

```sh
git clone https://github.com/ztripez/co_moe.git
cd co_moe
uv sync --frozen
uv run python replicate_granite_pair.py --output runs/granite-replication
```

This runs injection, donor-weight controls, and routed substitution with fixed
settings. It writes `receipt.json` and `activations.pt` to the chosen directory.
It references the included reciprocal and feedback receipts without rerunning
them. Only the selected Granite expert is instantiated for this experiment,
although its loader downloads the Granite checkpoint files.

Run the paired experiments separately if needed:

```sh
uv run python reciprocal_inference.py --output runs/reciprocal
uv run python feedback_inference.py --output runs/feedback
```

Run commands from the repository root. Choose a fresh output directory for each
run to preserve the recorded evidence under `results/`. The benchmark scripts
require a new output directory; some earlier scripts can overwrite their output.

## Experiment index

| Experiment | Executable | Recorded method and results |
| --- | --- | --- |
| Granite injection, weight controls, substitution | `replicate_granite_pair.py` | [Replication](GRANITE_PAIR_REPLICATION.md) |
| Reciprocal live inference | `reciprocal_inference.py` | [Reciprocity](RECIPROCAL_INFERENCE.md) |
| One ordered return path | `feedback_inference.py` | [Feedback](FEEDBACK_INFERENCE.md) |
| Fixed expert-0 prediction comparison | `benchmark_substitution.py` | [Prediction benchmark](SUBSTITUTION_BENCHMARK.md) |
| Half of layer-7 routed contributions | `benchmark_half_split.py` | [Three fixed masks](HALF_SPLIT_BENCHMARK.md) |
| Zero replacement at the same positions | `benchmark_half_zero.py` | [Matched-zero diagnostic](HALF_SPLIT_ZERO.md) |
| Historical DeepSeek injection / controls | `co_moe0.py` | [Injection](RESULT.md), [weight controls](WEIGHT_CONTROLS.md) |
| Historical DeepSeek expert imitation | `fit_native_expert.py` | [Bounded imitation](NATIVE_EXPERT_FIT.md) |
| Historical DeepSeek routed substitution | `substitute_expert.py` | [Live substitution](LIVE_SUBSTITUTION.md) |

The DeepSeek experiments are preserved as historical evidence. Their imitation
study did not show a pretrained-donor advantage over the parameter-matched
bypass. Its learned adapters are separate from the fixed-map Granite experiments.

## Evidence and report

`results/` contains the original numerical receipts, captured tensors, masks,
token losses, and plots. Tag [`v0.1.0`](https://github.com/ztripez/co_moe/tree/v0.1.0)
preserves the initial feasibility evidence;
[`v0.2.0`](https://github.com/ztripez/co_moe/tree/v0.2.0) adds the prediction
benchmarks, matched-zero controls, and first technical report. The historical
receipts and tags remain unchanged.

The public edition updates repository documentation and the report's access
statement without changing experimental results. See
[paper build instructions](paper/README.md) to render the report from recorded
evidence without running models.

Author: **Ztripez von Matérn**. This is an independent research prototype and a
first technical-report draft, with no assigned venue or claimed peer review.
