# Reciprocal live inference — PASS (OLMoE / Granite)

Two independently pretrained, fully resident MoE backbones completed native
generation while reciprocally consuming expert responses produced from each
other's ongoing inference. Suppressing either outgoing response changed the
other model's subsequent computation. Disabling exchange restored both baselines
exactly. No training or quality optimization occurred.

**This uses OLMoE and Granite, not a live OLMoE–DeepSeek pair.** The full OLMoE and
DeepSeek backbones exceed available RAM at the float32 precision used here,
while GPU free memory was about 2 GiB. A smaller second backbone kept the proof
local and avoided introducing quantization/offload infrastructure. A full live
DeepSeek backbone was not attempted.

## Models and independent streams

| | A | B |
|---|---|---|
| Checkpoint | `allenai/OLMoE-1B-7B-0924` | `ibm-granite/granite-3.1-1b-a400m-base` |
| Revision | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29` |
| Hidden width | 2048 | 1024 |
| Layers | 16 | 24 |
| Experts | 64, top 8 | 32, top 8 |
| Attention | 16 query / 16 KV heads | 16 query / 8 KV heads |
| Expert | Layer 7, expert 0 | Layer 7, expert 0 |
| Prompt | `The capital of France is` | `The largest planet is` |
| Prompt tokens | 5 | 4 |

The models have separate objects, tokenizers, inputs, KV caches and generated
histories. Two local Python threads keep both inference streams alive while
synchronizing once at a shared barrier. They are not separate OS processes or
distributed workers. Their prompts and token positions are not treated as aligned.
Granite also uses different residual, attention, embedding and logit scaling.
Both selected expert operators still use SwiGLU.

## The single coupled execution

Each native backbone runs through layer 7. Capture its actual normalized MoE
input at its own last prompt position, and evaluate its own pretrained expert 0
with unit router weight to produce an auxiliary expert response. The original
native MoE routing and computation still run unchanged.

- A sends `E_A(x_A)` (2048 values); B truncates it to 1024 and adds 0.01 times it
  to B's layer-7 residual at B's last prompt position.
- B sends `E_B(x_B)` (1024 values); A zero-pads it to 2048 and adds 0.01 times it
  to A's layer-7 residual at A's last prompt position.

These are expert responses from live source activations, not cached offline
responses or decoded text. Each expert uses its own source model's activation;
its result is foreign computation for the receiving stream. This experiment
exchanges expert results, rather than applying each expert directly to the other
model's activation. The earlier experiments separately established foreign-input
execution and execution-slot substitution.

Both source prefixes and response vectors exist before either recipient resumes
its coupled suffix. The saved event trace records that ordering. Each native
backbone then runs its remaining layers and generates four tokens with its own
cache. Both full backbones stay resident throughout. The expert-response
exchange occurs only during prefill, once per direction; it is additive, not an
expert-slot substitution. There is no learned adapter or norm matching.

## Causal observations

Five paired conditions ran: native baseline, both directions active, A→B
suppressed with B→A retained, B→A suppressed with A→B retained, and exchange
disabled again. This is ten completed generations, all with finite recorded
activations and logits.

Measurements use each recipient's last prompt position before generated
histories could diverge. The intervention zeros only the delivered expert response;
the source's live expert still executes.

| Suppressed direction | Recipient | Layer 8 Δ L2 vs coupled | Final logits Δ L2 vs coupled |
|---|---|---:|---:|
| B → A | OLMoE | 0.03300568 | 3.51766276 |
| A → B | Granite | 0.01075248 | 0.49266958 |

In each intervention the recipient's incoming contribution became exactly zero.
Suppressing a model's outgoing response left that model's own recorded downstream
activation and logits exactly equal to the fully coupled run, since the reverse
message was produced before either suffix resumed. Thus the two interventions
isolated the two causal directions in the same paired execution.

The fully coupled changes relative to native baselines had the same L2 values
as the table. Source expert inputs before exchange were exactly equal to native
inputs. Disabling exchange recovered both native downstream activations and
logits with maximum absolute error **0**, and identical generated tokens.

OLMoE generated ` Paris.\n\n`; Granite generated ` Jupiter, which` in every
condition. Unchanged greedy tokens do not imply unchanged internal computation
and are not a quality or acceptance metric. Activation scales differ between
models; their difference magnitudes should not be interpreted as comparable
cooperation strength.

## Receipt and limits

Run: `uv run python reciprocal_inference.py`

- [Executable](reciprocal_inference.py)
- [Receipt, model identities, event traces and interventions](results/reciprocal-inference/receipt.json)
- [Recorded source inputs, responses, delivered messages and recipient tensors](results/reciprocal-inference/activations.pt)

Runtime: one machine, CPU float32, 6 Torch threads per CPU operation, eager
attention/experts, seed 0, deterministic algorithms. All pretrained parameters
are frozen. Ruff passes; no broad suite was added or run. The first attempt
loaded both backbones but used OLMoE's module name for Granite's MoE hook and
stalled at synchronization; the corrected run uses Granite's actual
`block_sparse_moe` boundary. No numerical parameter was tuned in response.

This establishes reciprocal computational coupling of two live independent
inference streams for the tested OLMoE/Granite pair. It is one synchronous
exchange per direction, not a repeated feedback loop. It does not demonstrate a
live DeepSeek pair, semantic alignment, useful cooperation, quality improvement,
performance benefit, or interoperability between different expert activation
operators. Work stops at this feasibility milestone.
