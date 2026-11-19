# Paper tables and figures

`make_results.py` turns the output of the analysis scripts into the paper's tables and figures:

```bash
python paper/make_results.py <report dir>
```

It reads `means.csv`, `contrasts.csv`, `cost.csv` and `rpi.csv` from the report directory (see
[`docs/reproduce.md`](../docs/reproduce.md), step 6) and writes, next to this file:

- `tables/*.tex`: the LaTeX tables (main MSE per delay, primary contrasts, external methods, harm rate, ablations,
  RPI, memory sensitivity, cost);
- `fig3_delay.pdf`: MSE relative to the frozen backbone against the delay;
- `fig4_rpi.pdf`: revision predictability against the gain of VINE-S.

It also writes `contrasts_final.csv`, `rpi_cells.csv` and `rpi_hypotheses.json` into the report directory.
Nothing is selected or filtered: every complete setting is shown. The manuscript itself is not part of this
repository.
