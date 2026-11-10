Co-MoE first technical-report draft

Editable text: manuscript.md
Rendered report: ../output/pdf/co-moe-technical-report.pdf
Figures: figures/feedback.svg and figures/imitation.svg (PNG copies embedded in PDF)
Dense-substitution figure: ../results/half-split-benchmark/comparison.png
Matched-zero figure: ../results/half-split-zero/comparison.png

Build from the repository root:
  uv run --with reportlab --with matplotlib --with pypdf python paper/build.py

The renderer uses system DejaVu fonts (found through fontconfig).
It reads imitation chart values from git tag v0.1.0 and runs no model inference.
Section 8 uses substitution-benchmark, half-split-benchmark, and half-split-zero
artifacts. The paper and these additions are frozen in tag v0.2.0 alongside the
unchanged v0.1.0 evidence. The repository remains private.
The experimental receipts and historical implementations remain unchanged.
Author: Ztripez von Matérn. This is an unpublished first draft with no assigned
venue or listed affiliation.

REVIEW_REQUEST.md is a prepared outreach draft, not a record of a sent message
or external review. The separate Neural Flywheel/F0 papers are not part of
this Co-MoE report or tag.
