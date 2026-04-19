# Andreev 2007 dissertation — fig 29 / fig 30 data for 2·10⁶ and 4·10⁶ atoms

This file documents the two larger-model curves from Andreev §3.5 figs 29 +
30 that are **not** part of the TDMD T3 anchor-test run surface
(`config.yaml` fixes the model at 10⁶ atoms — see
`dissertation_reference_data.csv` for the primary gate).

Extracted alongside the 10⁶ curve during T6.0 (commit closing R-M5-8) from
the same scans (`docs/_sources/fig_29.png`, `fig_30.png`). Kept here as
future reference for:

- Scaling extrapolation sanity-checks when the T3 benchmark is re-scoped
  past v1.
- Cross-validating the 10⁶ curve's baseline extrapolation via the
  constant-work-per-atom invariant: `steps(N_atoms_1, N_procs) /
  steps(N_atoms_2, N_procs) ≈ N_atoms_2 / N_atoms_1` at matched N_procs.
- Quoting in papers / presentations without re-reading the scans.

## 2·10⁶ atoms

Middle curve of fig 29 (performance); middle red line of fig 30 assumed to
correspond based on dissertation §3.5 comm-overhead text (6–7 % → implied
efficiency ~93 %).

| n_procs | performance (steps/s) | efficiency (%) |
|---------|-----------------------|----------------|
| 1       | 0.0131                | 100.00         |
| 2       | 0.0252                | 96.00          |
| 4       | 0.0501                | 95.50          |
| 8       | 0.0995                | 95.00          |
| 16      | 0.1958                | 93.50          |
| 24      | 0.2898                | 92.20          |
| 31      | 0.3705                | 91.30          |

## 4·10⁶ atoms

Bottom curve of both figures. Dissertation §3.5 comm overhead 12–14 % →
asymptotic efficiency ~87 %, matches the bottom curve of fig 30 at high
N_procs.

| n_procs | performance (steps/s) | efficiency (%) |
|---------|-----------------------|----------------|
| 1       | 0.0072                | 100.00         |
| 2       | 0.0126                | 87.50          |
| 4       | 0.0252                | 87.50          |
| 8       | 0.0503                | 87.30          |
| 16      | 0.1001                | 87.00          |
| 24      | 0.1496                | 86.70          |
| 31      | 0.1935                | 86.80          |

## Extraction notes

- Y-axis reads on the middle/bottom curves are noisier (±~5 % rms vs ±~1 %
  for the top curve) because the PNG resolution shrinks the vertical
  separation between markers at low y-values.
- The single-rank value at `n_procs = 1` is back-fit from the asymptotic
  efficiency plateau (fig 30) combined with the fig 29 readout at
  `n_procs = 8`, not a directly readable data point — the curves' linear
  segment through the origin is extrapolated below the smallest visible
  marker (`n_procs = 2`).
- The 2·10⁶ assignment to the middle curve of fig 30 is an inference: the
  figure's legend only lists `4000000 атомов` and `1000000 атомов`, with
  the middle red trace understood as an interpolant / implicit 2 M from
  the §3.5 overhead text. If a future reader disputes this, the 10⁶
  primary CSV remains unaffected.
