# Paper figures

Run `python tools/build_paper_figures.py` from the repository root. Every figure is exported to `assets/figures/` as SVG (editable text), PDF (vector), and PNG (300 dpi). These are newly drawn implementation schematics, not extracted manuscript pages. They use no dataset imagery.

## Architecture

`architecture.{svg,pdf,png}`: IsoMamba uses three global encoder stages and a global bottleneck. Decoder stages at one-quarter, one-half and full resolution use four, three and two isolated experts, respectively. Additive feature skips connect corresponding scales. Encoder projection weights condition the router; encoder terminal states condition per-expert FiLM. A global refinement stage and convolution produce a residual added to the input. Stage widths are 32/64/128/256 and block counts are 4/6/6/8; refinement uses four global blocks.

Suggested caption: “Implementation overview of IsoMamba. Global encoder blocks establish image-level context. Decoder blocks route update-compatible tokens to separate recurrent experts, scatter their responses to the image grid, and coordinate them using encoder terminal states. Feature skips and the final image residual use addition.”

## Cluster–route–isolate

`cluster_route_isolate.{svg,pdf,png}`: The actual router applies detached encoder projection slices, directional alignment and learned fusion, plus low-rank adaptation, followed by Softplus and normalization. Pooled clustering and anchor matching establish expert labels. Top-1 routing gathers each group's tokens into a square padded grid; each expert uses convolution and four-direction selective scan, FiLM, and scatter. Illustrative colored tokens are schematic, not observed labels. The schematic does not claim that nonzero indexing propagates STE gradients; see the audit.

Suggested caption: “Cluster–route–isolate in the current implementation. A projection-derived update representation establishes instance-dependent token groups and consistent expert labels. Tokens are gathered into padded grids and processed by dedicated recurrent experts before restoration of spatial layout. Independent streams separate the groups while FiLM shares encoder context.”

## Raw hidden-state separation

`hidden_state_separation.{svg,pdf,png}` and `assets/data/hidden_state_separation.json` are built from existing `analysis_outputs/raw_hidden_state_separation/dataset_summary.json`. The data artifact contains portable aggregates, protocol notes and the SHA-256 of the original summary; no local image paths or private checkpoint files are included. Re-extract from the local results using:

```bash
python tools/build_paper_figures.py \
  --source-summary analysis_outputs/raw_hidden_state_separation/dataset_summary.json
```

Suggested caption: “Post-hoc raw hidden-state separation at `decoder_level3.attention.0` on UIEB (n=90) and LSUI (n=399). Bars show image-level means and whiskers show ±1 standard deviation after averaging the four directional scores per image. Shared and isolated fixed-expert conditions reuse identical expert-0 tokenwise operands; their contrast probes recurrence organization, including group-specific predecessor order. Full isolation additionally uses group-specific experts and is a descriptive reference. Logical packing excludes dummy padding. All conditions derive from one trained checkpoint, rather than separately trained baseline networks.”

The separation score is between/(between+within+epsilon) on normalized raw recurrent states. Groups are defined by the model's routing labels, so larger scores indicate agreement with that partition. The plot is not a reconstruction-quality comparison or evidence of a trained shared-state baseline.

Qualitative enhancement panels, hidden-state response maps, routing maps and reconstruction-residual panels are not fabricated. Producing them requires reviewed checkpoint/baseline pairs, matching normalization and approved example images.
