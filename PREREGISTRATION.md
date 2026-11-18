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
| 2026-11-14 (18:44 KST, commit 230b25a) | **`vine_s` and `vine_lam` become co-primary (before any test result on ETTm2, Weather, Exchange or any later stream; ETTm2 had finished tuning but had no test row when this was committed).** Both are already in the grid and unchanged: `vine_s` as defined in the row below; `vine_lam` exactly as run, with its memory γ ∈ {0.995, 0.999, 0.9998} still chosen on validation (the γ = 0.9998 fix of the row below applies to `vine_s` and its controls only). **Primary contrasts:** each method vs `dric_l8`, and each method vs its label-only control (`vine_s` vs `vine_s_lo`; `vine_lam` vs `mc_ls_pr`, the same-feature label-only learner with the same memory range). **Multiplicity:** in every primary test family (each stream × H × delay, and each pooled test), the two methods' p-values for the same contrast are Holm-corrected together, on top of the existing within-table Holm correction. H1–H7 are evaluated separately for each method with unchanged criteria, and the ≥ 2-of-3-streams gate is applied to each. Both methods are always reported side by side on every stream; no per-stream choice between them is made after seeing test results, and no combination of them is added. | **Development-stream server results (ETTh1, ETTh2: 12 cells, 3 seeds, 4 delays; disclosed as development).** `vine_s` beat `dric_l8` in 14 of 16 seed-averaged settings (ETTh2 −1.7% to −9.4%, every seed; ETTh1 H = 96 −0.2% to −0.8%; ETTh1 H = 24 within ±0.35%), but `vine_lam` beat `vine_s` on ETTh2 in 7 of 8 settings (by 1.6% to 5.5%). On ETTh1 and ETTm1, `vine_lam` trails `dric_l8`, where `vine_s` does not. Neither variant dominates, so both are carried as co-primary with a multiplicity correction rather than choosing one from development test results. |
| 2026-11-14 | **Paper 2 headline changed to `vine_s` (before any result on ETTm2, Weather, Exchange or any later stream).** `vine_s` = VINE-λ (`vine_lam`) stacked on the validation-tuned DRiC-L8: the top layer treats DRiC's corrected forecasts as its backbone and learns from *their* revisions (`tsl/online/methods/stack.py`). The memory of the VINE family and of its same-feature label-only controls is fixed at γ = 0.9998 (half-life ≈ 3,466 steps, matching DRiC's published 128-block half-life at H = 24); every other knob is still tuned on validation with ≤ 24 configurations. New rows: `vine_s` (headline), `vine_s_lo` (control: same stack, base, features and memory, label-only MC-LS-PR top; isolates revision learning), `vine_s_pure` (λ = 1 top), `vine_lam_lm` (top layer without the base). `safe_mix` becomes an ablation. **Primary contrasts:** `vine_s` vs `dric_l8` (the strongest baseline) and `vine_s` vs `vine_s_lo` (effect of revision learning). H1, H2, H4–H7 are evaluated for `vine_s` with unchanged criteria; H3 uses the `vine_s` − `vine_s_lo` gain. Wide streams (C > 300, where DRiC-L8 is skipped) will use DRiC-G as the base; that variant is defined before any wide-stream run. Also added, not in the method: calendar-phase features (`k_phase`, no gain on development streams, left off) and a revision-hinted delayed hedge (`safe_mix` `hint="nowcast"`, not used: independent review found its weights could escape the fixed-share floor, since fixed, and its cost is prohibitive on wide streams). | **Decided on development streams only** (ETTh1, ETTh2, ETTm1: seen validation and test results, disclosed as development). (i) Tuning bias: the 5% validation stream cannot identify long memories, so tuning picked short memories (γ = 0.995) at d = H. The same learner with long memory was far better on development streams (e.g. vine_lam on ETTh1 H = 24: 0.761 vs 0.812). DRiC's long published default was always in its search space. (ii) An ablation showed that most of DRiC's advantage on ETTh1 comes from its per-phase learners and long memory, not from features. (iii) Stacked long-memory VINE-λ beat DRiC-L8 in all 10 development comparisons with a DRiC run (seed 0, H ∈ {24, 96}, d ∈ {H, 8H}; −0.03% to −10.3%). Against the label-only stacked control it won 13 of 16 comparisons (seeds 0 and 1), tied 1 and lost 2 (by 0.02% and 0.09%). Confirmation rests on the untouched streams. Independent code review of the new code: no leaks; all findings fixed. |
| 2026-11-14 | **Additions to the `vine_s` analysis (still before any confirmation-stream result).** (a) Memory sensitivity: `vine_s_g9995` and `vine_s_g9999` take each cell's tuned `vine_s` configuration with γ = γ_a ∈ {0.9995, 0.9999}. They are reported in a sensitivity table and never used to select anything. Disclosure: γ = 0.9998 matches DRiC's published half-life only at H = 24; at H = 96 it is about 3.5× shorter, and DRiC's own half-life is tuned on validation. (b) H3 gets a second, theory-backed correlation: validation RPI of the DRiC-corrected forecasts vs the `vine_s` − `dric_l8` gain, reported alongside the pre-registered `vine_s` − `vine_s_lo` one (`methodology/paper2_theory_stacked.md`, S9–S10). (c) Diagnostics reported for `vine_s`: base-drift share of revisions, measured by replaying with each DRiC update deferred one step (S4–S5); and overshoot in the recovery plots (double correction after shifts). (d) Leak test: the future-perturbation test now also runs `vine_s` and `vine_s_lo` with inputs different from targets. | Theory review of the stacked method (`methodology/paper2_theory_stacked.md`, "risks a reviewer will raise"). |
| 2026-11-13 | **Paper 1 reframing (before any real-data SOMA result).** The memory-only SOMA is close to IGMN/GMR. Paper 1's claims are restricted to what that family lacks: (a) statistics in a variable-agnostic, instance-normalised window space, so variables can be added and removed and new ones forecast zero-shot; (b) a bounded budget with retention-cost merging plus a nursery, against IGMN's unbounded growth; (c) time-synchronous invariance, stated for exact arithmetic and at common endpoints only, together with the spectral forgetting law. Mandatory additions: IGMN and the K = 1 ridge; invariant-by-construction learners (single-pass SGD, an analytic RLS head); equal-bytes ER/DER++ at 1× and 5×; joint retraining to convergence and batch-EM GMR as upper bounds; EWC/SI with decay γ < 1; absolute zero-shot error on new variables; paired removal counterfactuals; prequential RER; a K_max sweep in which merges actually fire. The bounded residual stays in the method only if, after fair ablations on first-period validation, it improves AE; otherwise it is removed. Config `gap` = L + H. SynthRegimes (sync variant as the D0 headline) is confirmatory, not independent, because the spectral nursery seeding was designed on it. | Independent SOMA review |

| 2026-11-15 (before any DSOF or Proceed result on any segment) | **External leak-free baselines DSOF (ICLR 2025) and Proceed (KDD 2025) added.** Both are ported from the official code (github.com/yyalau/iclr2025_dsof; github.com/SJTU-DMTai/OnlineTSF) to the truthful protocol in `code/tsl/online/methods/{dsof,proceed}.py`; the module docstrings list every deviation (DSOF: TD origin and teacher shifted by the point lag ℓ, student trained offline on the frozen pretrained backbone; Proceed: newest complete sample = origin t − d, no offline pass over validation labels). Each starts from its published configuration and is tuned on validation like every other method (same budget 24, same RNG; search spaces in `code/scripts/online_grid.py`). Leak-checked by the future-perturbation test (`code/tests/test_dsof_proceed.py`). They join the baseline table and the "best external method" comparison as secondary contrasts (each co-primary vs each of them, Holm within the table). Primary contrasts, co-primary methods and H1–H8 are unchanged. Finished cells are completed with `--methods dsof proceed` (other methods' tuning and test rows untouched; their search-space hashes do not change). FSNet and OneNet are not rerun: both were designed under the leaky protocol, DSOF and Proceed each re-evaluated them leak-free and found them dominated, and the 2026 delay-aware audit covers them; `ogd_head` and `periodic` represent gradient-based adaptation. | Reviewers will ask for the two most recent methods built for delayed feedback under the same protocol, not as citations. Decided before any of their results exist. |
| 2026-11-16 (after the first full analysis run on the development streams and ETTm2) | **How the two pre-registered tests are combined.** The plan names a block bootstrap (B = 1000) and a Diebold–Mariano test with HAC variance, with Holm within each table, but not how the two are combined. The first full run (`code/scripts/vine_report.py`) used the larger of the two p-values. That rule can never reject: a bootstrap p-value cannot fall below 1/B = 0.001, so after Holm over the 120-test primary table no contrast can reach 0.05, whatever the data. Corrected rule: a contrast is significant when its Holm-adjusted DM p-value is below 0.05 **and** its bootstrap 95% CI excludes zero. Holm families, B, block lengths, contrasts and hypotheses are unchanged. | Mathematical defect of the combining rule, independent of the data. **Disclosure:** this change was made after seeing the first run's output, in which nothing was significant. Under the corrected rule, the large ETTm2 and ETTh2 gains of `vine_s` are still **not** individually significant; the paper reports this. |
| 2026-11-16b (after RPI on development cells only: ETTh2 H = 24 seed 0 at all four delays and ETTh1 H = 24 seed 0 at d = H; before any RPI on ETTm1, ETTm2, Weather or Exchange and before any RPI–gain correlation) | **RPI estimator fixed.** The plan defines RPI as the out-of-sample R² of a ridge regression of the revisions on φ_s on validation, but fixes neither the split nor the penalty. Fixed now: per-channel ridge on standardised features (the `vine_s` top layer's feature map, built from the DRiC-L8 forecasts F), fitted on the first half of the validation stream and scored on the second half (the tuning split), pooled over channels, averaged over h = 2..H; the penalty is chosen from {1, 10, 100, 1000} by blocked 5-fold CV inside the first half. RPI is computed for B (frozen), F (DRiC-L8, pre-correction) and G (`vine_s`, post-correction). H3 (both correlations, amendment 2026-11-14 b) uses RPI(F) with seed-averaged cells (dataset, H, d). H6 compares RPI(G) with RPI(F); because the ratio is undefined when RPI(F) ≤ 0, H6 is evaluated on cells with RPI(F) > 0 and the medians of both are reported. Nesting defect: out-of-sample R² of ε_{s+1,h−1} on φ_s, with ε the residual of F's error regressed on φ_s. **Exploratory, labelled as such:** an online (prequential) RPI on the same rows with an expanding-window refit, added because these development cells showed a fixed-split RPI that was negative on ETTh2 (the fitted relation did not carry over between the halves) while an online refit was positive. Code: `code/scripts/vine_rpi.py`. | Under-specified diagnostic; disclosed that the online variant was added after seeing these development cells. |
| 2026-11-18 (after the confirmation results on ETTm2 and Exchange) | **Exploratory, not pre-registered: persistence fallback.** Because last-value persistence beat every corrector on Exchange, a causal per-horizon switch between `vine_s` and `persist_last` (lower discounted squared error on fully matured origins, origin <= t - d; discount = `vine_s` memory 0.9998; nothing tuned; `vine_s` until the first label matures) is replayed exactly from the stored per-origin errors. Statistics as in `vine_report.py`, Holm within its own table. It does not enter any hypothesis, co-primary contrast or headline. Code: `code/scripts/vine_fallback.py`. | Added after seeing confirmation results; reported only as exploratory. |
