# Pre-registration (frozen before any real-data result is inspected)

Registered: 13 November 2026.

**What had been seen when each part was written.** Part 1 (protocol, hypotheses, gate) was written before any real-data
result of VINE existed. Only synthetic results (`tests/test_vine.py`) and the 3-epoch smoke run in `code/README.md` had
been seen. The amendments below were then made on the same day, after seeing three kinds of result:

- validation-segment results on ETTh1, ETTh2 and ETTm1 (DLinear, H = 24);
- one untuned **test-segment** pilot on ETTh2 (DLinear, H = 24, seed 0, default hyperparameters);
- a 2-epoch smoke run on ETTh1, including its test segment.

Because those streams influenced the design, the week-4 gate also requires the result on design-untouched streams
(ETTm2, Weather, Exchange). Every table will mark ETTh1, ETTh2 and ETTm1 as **development streams**. The backbone's
early stopping uses the same validation segment as corrector tuning, and the paper discloses this.

Any change after this date is logged in the amendment table at the bottom, with its date and reason, **before** the
affected experiments run. Results that contradict a hypothesis are reported, not dropped.

## Paper 2 (VINE): online correction under truthful label delay

### Protocol (amended from the methodology after the prior-art audit of 13 Nov 2026)

- **Label release.** The label block Y_s = y[s+1..s+H] is released at s+d, with d ≥ H. Equivalently, each value
  y_τ is released at τ+ℓ, where the **point lag** is ℓ = d − H. All methods may read released values through the
  guarded history (this is the audit's "prefix reduced accordingly" allowance). We report results by ℓ:
  d ∈ {H, 2H, 4H} gives ℓ ∈ {0, H, 3H}, and the sweep figure extends to d = 8H.
- **Splits and segments.** FSNet split 20:5:75. The scaler is fitted on the training rows only.
- **Tuning.** All tuning happens on the 5% validation segment, run as a mini online stream under the same protocol.
  The test segment is touched once per configuration that is frozen in a config file.
- **Hyperparameter budget.** At most 24 configurations per method, using the same random-search seed for every
  method, VINE included. Published defaults are always one of the 24.
- **Seeds and backbone.** Seeds 0, 1 and 2 (backbone training seed). Every corrector in a cell shares the same frozen
  checkpoint.
- **Statistics.**
  - Paired per-origin loss differentials, circular moving-block bootstrap with block length ≥ max(2 × season, 2H)
    (because H-step targets overlap), B = 1000.
  - Diebold–Mariano test with HAC variance (bandwidth ≥ H−1).
  - Holm correction within each table family.
  - Across cells: Wilcoxon test on per-cell relative gains.

### Baselines (all run under the identical protocol; faithful ports are bit-identical to their published code)

| Group | Methods |
|---|---|
| Reference | frozen, persistence (seasonal naive), delayed OGD (head and full), periodic retrain |
| Faithful closed-form ports | audit RLS-Bank (`rls_audit`); DRiC-L8 and DRiC-G (`phases=all` as headline; single grid scored on grid origins as in the original paper); EPOC |
| Closed-form reimplementations | ELF-style (ELF-mix) |
| Same-feature label-only learners (isolate revision learning) | MC-LS (block-matured); MC-LS-PR (per-row maturity at the point lag, EFC-style use of partial errors) |
| Naive revision regression (mandatory ablation) | Matured errors regressed on revisions, without the fixed point |
| Combiners | AdaHedgeD-style delayed Hedge over the same experts; SAFE-MIX restricted to label-only experts |
| Deep online adapters (laptop: narrow streams) | Proceed, DSOF, TAFAS, COSA, PETSA, ADAPT-Z |

### Hypotheses and success criteria

- **H1 (headline).** At ℓ ≥ H (that is, d ≥ 2H), VINE+SAFE-MIX has lower cumulative MSE than the best of
  {rls_audit, ELF-mix, DRiC, EPOC, MC-LS, MC-LS-PR, persistence} in at least 60% of (dataset, backbone, H) cells, with a
  Holm-significant median gain. At ℓ = 0 it is no worse than the best of them.
- **H2 (no harm).** On ETTh2, VINE+SAFE-MIX is non-inferior to frozen at every d ≤ 8H, with a margin of +1% MSE
  (one-sided bootstrap CI). Frozen is one of the experts, so this is close to guaranteed by design. The informative
  contrast is SAFE-MIX against SAFE-MIX restricted to label-only experts (`safe_mix_lo`).
- **Headline row.** `safe_mix` is the single headline method. Every other VINE variant is an ablation and is never
  chosen per cell. The primary contrasts are VINE vs MC-LS-PR and safe_mix vs safe_mix_lo.
- **Seeds.** The seeds are backbone-training seeds; the correctors are deterministic. Cross-cell tests treat
  cells from the same dataset as dependent: Wilcoxon tests run on per-dataset median gains.
- **H3 (when it pays).** The gain over MC-LS-PR correlates positively with the validation Revision Predictability
  Index (Spearman ρ ≥ 0.5).
- **H4 (consistency).** The DSOF-style teacher target is worse than VINE in at least 70% of cells.
- **H5 (delay decoupling).** VINE's slope of MSE against d is at most half of MC-LS's.
- **H6 (efficiency).** The post-correction RPI is at most 25% of the pre-correction RPI.
- **H7 (cost).** At H = 24, VINE's median per-step latency (update + predict) is at most 2× rls_audit's on the same
  machine, and its learned state (float32-equivalent) is at most 12 kB per channel. Pending queues held by the
  harness are charged to each method when state is compared.
- **H8 (foundation-model backbones).** VINE improves TTM-r2 forecasts under the same protocol.

### Gate (week 4)

The gate passes if H1's direction holds on at least 2 of the 3 ETT streams at d ≥ 2H **and** H2 holds on ETTh2.
Otherwise we switch to fallback F1: a paper built on protocol, theory and the RPI diagnostic.

## Paper 1 (SOMA)

Hypotheses H1–H9 and gates G1 and G2 are as in `methodology/paper1_continual_tsf_methodology.md` §8 and §11.4, with
these amendments from the prior-art audit:

- IGMN/GMR (incremental Gaussian mixture regression) and SyMPLER are added as baselines.
- Joint-retraining equivalence is presented as a lemma, not a headline claim.
- Taskification invariance is stated precisely: it holds for boundary placement given the stream order, not for
  reordering the stream.

## Amendments

| Date | Change | Reason |
|---|---|---|
| 2026-11-13 | Delay reported by point lag ℓ = d − H. VINE's anchor learns at the point lag. MC-LS-PR, persistence, the naive revision regression and AdaHedgeD-style combiner baselines added. | Prior-art audit: EFC (arXiv 2610.22872) shows partial errors are legal under block release; the 2026 audit's bar is ELF-mix plus persistence |
| 2026-11-13 | New options for VINE **and** the same-feature baselines: lookback-block DCT features (`k_lb`), a DRiC-style shrinkage gate (`gate`), horizon-DCT projection (`k_h`), and memory up to γ = 0.9998. Also added the TD(λ)-style variant `vine_lam`, which blends revision targets with per-row matured labels, with λ tuned on validation. Search spaces widened accordingly (`scripts/online_grid.py`). | Decided on **validation segments only** (ETTh1, ETTh2 and ETTm1 with DLinear; no test data). The base VINE features were under-regularised relative to the faithful DRiC/EPOC ports. At ℓ = 0 revision targets add bias without a staleness benefit, which is the I_T term of Theorem 1. |
| 2026-11-13 | Test protocol: every method starts online at the beginning of the validation segment and is scored on the test origins only. | Identical for all methods. Removes cold-start artefacts from 5%-length streams. The test segment is still never used for tuning. |
| 2026-11-13 | H1 is now framed as conditional: VINE's advantage is predicted to grow with the point lag ℓ and with the validation RPI. H1's criteria are unchanged and still evaluated as written. | Validation evidence at ℓ = 0, logged before any tuned test result. |
| 2026-11-13 | Independent pre-grid review: critical bug C1. The whole-matrix ρ_max rescale fired on almost every solve and damped any carried correction by 0.98^(h−1). It is replaced by per-singular-value clipping of the whitened transition (ρ_max = 1.02), refreshed at every solve and active only after 2p revision samples. Ridge now shrinks each row's new information towards the propagated previous row. An explicit propagation discount κ is introduced (w_h = G_h + κ w_{h−1} M; κ = 0 is the teacher target, κ = 1 the consistent fixed point), default 0.9, chosen on the **synthetic** DGP only and tuned on validation. The grid launched earlier is quarantined (`results/online_grid_INVALID_stabiliser_bug`) and its partial results were not inspected. | Review findings C1 and M6. After the fix, the synthetic DGP shows the TD-style bias–variance trade-off: κ = 1 and κ = 0 are both worse than κ ≈ 0.9. |
| 2026-11-13 | Tuning scored after a burn-in (second half of the validation stream); configurations tuned **per delay** (8H uses the 4H configuration); tuning files carry a hash of the search space. Baselines added: safe_mix_lo, single-grid DRiC (as published), delayed OGD (head) and periodic retraining. Per-step median and p95 latency recorded. | Review findings M1, M2, M3 and M5 |
| 2026-11-13 | κ discounts only the non-intercept directions (M′ = diag(1, κ, …, κ) M). VINE's non-intercept features are centred by a running γ-mean, so a constant backbone bias propagates to every horizon undamped. Regression test `test_constant_bias_propagates_to_long_horizons`. Grid relaunched with tuning-cache version 2026-11-13c; the earlier relaunch was stopped during backbone pretraining, before any corrector result. | A uniform κ reintroduced reviewer finding C1 (bias decayed as κ^(h−1)). Decided on synthetic data and the unit test only. |
| 2026-11-13 | **Paper 1 reframing (before any real-data SOMA result).** The memory-only SOMA is close to IGMN/GMR. Paper 1's claims are restricted to what that family lacks: (a) statistics in a variable-agnostic, instance-normalised window space, so variables can be added and removed and new ones forecast zero-shot; (b) a bounded budget with retention-cost merging plus a nursery, against IGMN's unbounded growth; (c) time-synchronous invariance, stated for exact arithmetic and at common endpoints only, together with the spectral forgetting law. Mandatory additions: IGMN and the K = 1 ridge; invariant-by-construction learners (single-pass SGD, an analytic RLS head); equal-bytes ER/DER++ at 1× and 5×; joint retraining to convergence and batch-EM GMR as upper bounds; EWC/SI with decay γ < 1; absolute zero-shot error on new variables; paired removal counterfactuals; prequential RER; a K_max sweep in which merges actually fire. The bounded residual stays in the method only if, after fair ablations on first-period validation, it improves AE; otherwise it is removed. Config `gap` = L + H. SynthRegimes (sync variant as the D0 headline) is confirmatory, not independent, because the spectral nursery seeding was designed on it. | Independent SOMA review |

