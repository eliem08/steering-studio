# Steering Studio — Product Roadmap

## Product outcome

A cautious research sandbox for comparing activation interventions, controls, and capability regressions across supported transformer models.

## Users and problem

- Mechanistic interpretability researchers exploring representation interventions.
- Model engineers testing hypotheses about internal features.
- Educators demonstrating intervention mechanics.

## Current foundation

- GPT-2-style activation-difference direction, layer hook, scale slider, baseline/intervention comparison, saved JSON, dashboard and Colab notebook.

## Roadmap

### Phase 1 — Safe experimental controls
- [ ] Validate model architecture and target layer before execution.
- [ ] Add zero-scale, random-direction, positive/contrast swap, and sham controls.
- [ ] Sweep scales/layers and save a matrix of results.
- [ ] Add task-level capability regression suite, not just qualitative generations.
- [ ] Log activation norms, prompt sets, exact model revision and intervention config.

### Phase 2 — Research-grade steering
- [ ] Add directions estimated from multiple prompt examples with train/test split.
- [ ] Add hook adapters for additional architectures only with dedicated tests.
- [ ] Add token-position and generation-step controls.
- [ ] Compare intervention effects across datasets and report uncertainty.
- [ ] Add experiment export and side-by-side output viewer.

### Phase 3 — Final product
A local interpretability lab for intervention sweeps and controls, with capability regression dashboards and reproducible experiment artifacts.

## Safety and validity boundary

Steering is not guaranteed to produce a stable semantic attribute or preserve other capabilities. The final product must foreground control conditions and task regressions rather than market a universal model-control knob.
