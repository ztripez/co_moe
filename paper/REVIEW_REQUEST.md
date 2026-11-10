# Independent review request — draft, not sent

Recipient: Professor Richard Johansson, University of Gothenburg.
Email: richard.johansson@gu.se
Contact and research area checked on 10 November 2026 against his
[university profile](https://www.gu.se/om-universitetet/hitta-person/richardjohansson).

Attach `output/pdf/co-moe-technical-report.pdf`. The repository remains private;
this draft does not grant access or claim that the recipient has reviewed it.

Subject: Feedback request: cross-model MoE expert substitution

Dear Professor Johansson,

I'm Ztripez von Matérn, an independent developer working outside academia. I've
written my first technical research draft and would appreciate a brief critical
reading or a pointer to relevant prior work.

The study, Co-MoE, connects independently pretrained OLMoE and Granite models at
internal expert boundaries. With frozen weights and fixed dimensional maps, the
experiments demonstrate foreign-expert injection, native-routed substitution,
reciprocal exchange, and one causally ordered return path. A denser substitution
test also completes, but worsens reference-text prediction. Zeroing the identical
positions is less damaging in all three tested masks. The report does not claim
useful knowledge transfer or improved model capability.

The questions I'd most value your perspective on are:

1. Is the distinction from model stitching and existing model-composition work
   meaningful, or am I overlooking a closer precedent?
2. Do the interventions support the stated execution and causal-dependence
   claims, or is there a missing explanation or control?
3. Would this be worth developing as a technical report or workshop submission?

I've attached the draft. The code and recorded artifacts are frozen in a private
repository, and I can provide access if helpful. Even a short comment or a
suggestion of someone better suited to assess it would be appreciated.

Best regards,
Ztripez von Matérn
