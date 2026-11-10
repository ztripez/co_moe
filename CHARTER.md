# Co-MoE research charter

Discover what is possible, then what is useful, then what is better.

| Evidence level | Research question | Success condition |
|---|---|---|
| 1. Computational coupling | Can foreign computation enter another model? | Causal influence demonstrated |
| 2. Functional substitution | Can a foreign expert occupy a native expert's execution slot? | Actual routed substitution completes |
| 3. Cooperative inference | Can two independent model computations influence each other? | Bidirectional causal influence demonstrated |
| 4. Feedback | Can A influence B and return through B to influence A? | One causally ordered round trip demonstrated by a first-message intervention |

Functional substitution does not require functional equivalence. The native
router must select the target expert, the foreign expert must execute in its
place under the original mixture weight, and the receiving model must continue.
Finite computation, downstream causal effects, and native restoration supply
execution evidence. Matching tokens, low imitation loss, and improved output
quality are not prerequisites.

- Completion before quality: establish that the proposed computation executes.
- Measurement is evidence, not an optimization objective. Record numerical
  differences to establish causality, not to maximize them.
- No benchmark-driven development. Do not tune prompts, adapters, experts,
  datasets or coefficients for better scores unless explicitly assigned that
  functionality or optimization question. Selecting a context solely to ensure
  the target router slot is exercised is an execution requirement.
- Negative quality results do not invalidate feasibility. Worse inference can
  still demonstrate that foreign computation participates.
- Stop once the assigned hypothesis is demonstrated. Do not automatically move
  to the next evidence level or begin hardening or optimization.

The simplest model-independent bridge already works mechanically for the tested
pair. The current obstacle is transfer of useful learned computation, rather than
interoperability of execution. This observation is bounded to the demonstrated
architectures and expert operators; the original OLMoE/DeepSeek experts use SwiGLU and matching input/output
widths. The later OLMoE/Granite pair retains SwiGLU but has different widths.

Expert imitation and donor-weight controls address separate questions about
usefulness and learned structure. They do not define acceptance for execution
feasibility. The completed bounded imitation run supplies no evidence of a
pretrained-donor advantage over the parameter-matched bypass.

Current achieved level: one causally ordered A -> B -> A feedback cycle for
OLMoE/Granite, isolated by suppressing the first message; see
[FEEDBACK_INFERENCE.md](FEEDBACK_INFERENCE.md). No sustained-loop, stability or
usefulness claim is made. The preceding rungs remain documented in
[RESULT.md](RESULT.md), [LIVE_SUBSTITUTION.md](LIVE_SUBSTITUTION.md), and
[RECIPROCAL_INFERENCE.md](RECIPROCAL_INFERENCE.md). Live OLMoE/DeepSeek reciprocity
and feedback remain untested. All four rungs are feasibility results; useful
learned transfer is not established. Work stops here; no further experiment,
optimization or paper-writing campaign starts automatically.
