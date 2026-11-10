# One closed-loop exchange — PASS

One message originating in OLMoE's ongoing inference changed Granite's later
computation, and a freshly computed response from that changed state returned to
change OLMoE. A first-message suppression intervention isolated the complete
**A → B → A** causal path. This is one closed-loop feasibility experiment, not
an optimization or sustained feedback campaign.

Run: `uv run python feedback_inference.py`

## Unchanged experimental ingredients

A: `allenai/OLMoE-1B-7B-0924`, revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
B: `ibm-granite/granite-3.1-1b-a400m-base`, revision
`408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`.

The pair, prompts, two auxiliary experts (layer 7 expert 0 in each model),
zero-padding/truncation maps, coefficient 0.01, CPU float32, deterministic seed 0,
eager implementations, and four generated tokens per stream are unchanged from
the reciprocal experiment. No additional experts, model family, training,
learned adapters, or distributed execution were introduced. Both full backbones
remain resident, with separate tokenizers, prompts, KV caches and generation
histories in two local threads.

## Two rounds with causal ordering

1. **Round 1, layer 7:** A computes its existing expert 0's response on its live,
   normalized layer-7 MoE input. B truncates the 2048-value response to 1024 and
   adds 0.01 times it at B's own last-prompt residual position. A receives nothing
   in this round.
2. **Native computation:** B continues through layer 8 and layer 9. Its normalized
   layer-9 MoE input is produced after the first injection. A continues natively
   through layer 9 without an incoming message.
3. **Round 2, layer 9:** B reuses its **same layer-7 expert 0** on the later live
   layer-9 input to compute a new return response. A zero-pads this 1024-value
   response to 2048 and adds 0.01 times it at A's layer-9 residual output.
4. Both streams continue native inference and generation. A's observed downstream
   activation is at layer 10, after receiving the return.

There is a local barrier in each round. The event trace verifies the strict
order: A's first message ready → B's first injection → B's return message ready
→ A's return injection. The return uses B's layer-9 input, not its old layer-7
activation or an offline recorded response. It depends on B's intervening native
computation; it does not claim dependence on layer 9's MoE output itself.

## Intervention and observed chain

Four paired conditions ran: native baseline, closed loop, first-message
suppressed, and exchange disabled again. The intervention zeros **only the first
message delivered from A to B**. A still computes its original response, and B
computes and sends a fresh return in both active conditions. Consequently the
intervention retains the return path rather than disabling both exchanges.

All deltas below compare closed-loop execution with that intervention at the
same last-prompt position, before generated token histories can diverge.

| Observation | Measured difference |
|---|---:|
| A's original first expert response, max absolute | **0** |
| A's layer-9 residual before receiving the return, max absolute | **0** |
| Delivered first message, L2 | 1.52869594 |
| B's later layer-9 normalized MoE input, L2 | 0.47161409 |
| B's freshly computed return response, L2 | 0.02961485 |
| A's received return response, L2 | 0.02961485 |
| A's residual immediately after return injection, L2 | 0.00029615025 |
| A's downstream layer-10 activation, L2 | **0.00031965604** |
| A's final last-prompt logits, L2 | **0.02700368** |

A's state immediately before reception is exactly unchanged while the return
and subsequent A computation change. Together with the ordered trace and changed
B intermediate state, this attributes the returning effect to A's first message
passing through B. The magnitudes span different tensor spaces and are evidence
of propagation, not comparable quality scores.

All four paired conditions completed: eight finite recorded generations, two
prefill pause points per stream, and one extra call to each existing expert in
each active condition. No exchanges occur during subsequent decode steps.
OLMoE generated ` Paris.\n\n`; Granite generated ` Jupiter, which` in all
conditions. Token agreement was incidental.

Disabling exchange restored both models' recorded layer-10 activations and
last-prompt logits with maximum absolute error **0**, and restored identical
generated tokens. The trace-order checks passed in both active conditions.

## Evidence and scope

- [Executable](feedback_inference.py)
- [Receipt, exact settings, model identities and ordered event traces](results/feedback-inference/receipt.json)
- [Source activations, messages, later return inputs and downstream tensors](results/feedback-inference/activations.pt)

Ruff passes. No broad suite, benchmark, extra training, parameter tuning, or
quality optimization was performed. Earlier receipts remain intact.

This establishes **one causally ordered round trip** across the existing
OLMoE/Granite pair. It does not establish sustained-loop stability, semantic
alignment, useful learned transfer, output improvement, or live DeepSeek
feedback. Both expert operators still use SwiGLU. The receiving backbones use
native routers; the auxiliary messages are additive at the specified boundaries.

The four feasibility rungs now have executable evidence: injection, routed
substitution, reciprocal live exchange, and one feedback cycle. This experiment
ends here. Paper writing, additional scope and optimization have not started.
