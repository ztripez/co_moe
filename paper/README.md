# Co-MoE technical report

Author: **Ztripez von Matérn**. This is an unpublished first technical-report
draft with no assigned venue, listed affiliation, or claimed external review.

- [Editable manuscript](manuscript.md)
- [Rendered PDF](../output/pdf/co-moe-technical-report.pdf)
- [Public repository](https://github.com/ztripez/co_moe)

## Build

Run from the repository root after cloning with the release tags available:

```sh
uv run --frozen --with reportlab --with matplotlib --with pypdf python paper/build.py
```

The renderer uses system DejaVu fonts, located through `fontconfig` (`fc-match`).
It reads the frozen imitation receipts from Git tag `v0.1.0` and runs no model
inference or training. Its output is `output/pdf/co-moe-technical-report.pdf`.

The feedback and imitation figures are written to `paper/figures/`. The dense
substitution and matched-zero figures come from
`results/half-split-benchmark/comparison.png` and
`results/half-split-zero/comparison.png`.

## Evidence versions

Tag `v0.1.0` preserves the original feasibility and imitation evidence. Tag
`v0.2.0` preserves the first report and the subsequent prediction benchmarks and
matched-zero diagnostic. The public edition changes the repository-access
statement only; the experimental findings, receipts, and historical tags remain
unchanged.
