# Herschel–Bulkley fitting by the Mullineux determinant method

Python and browser implementations of

> G. Mullineux, "Non-linear least squares fitting of coefficients in the Herschel–Bulkley model," *Applied Mathematical Modelling* **32** (2008) 2538–2551. https://doi.org/10.1016/j.apm.2007.09.010

The Herschel–Bulkley model relates shear stress to shear rate, `y = y0 + K x^n` (yield stress `y0`, consistency `K`, flow index `n`). Fitting it is a non-linear least-squares problem. Mullineux showed that the normal equations are linear in `y0` and `K` with coefficients depending only on `n`, so they have a non-trivial solution only where a 3×3 determinant `F(n)` vanishes. `n` is therefore found first as a root of `F`, by bisection with no derivatives and no starting guess, and `y0` and `K` follow from a 2×2 linear solve. The paper also shows why the common shortcut — guess the yield stress, then fit `ln(y − y0)` against `ln x` — is unreliable.

## Try it in the browser

The fitting app needs no installation — it is a single HTML file that runs entirely in your browser (nothing is uploaded anywhere):

- **Live app:** https://ktwyw.github.io/herschel-bulkley-determinant-fit/web/herschel_bulkley_fit.html (served by GitHub Pages from this repository)
- **Without Pages:** open the same file through the GitHub HTML preview, https://htmlpreview.github.io/?https://github.com/ktwyw/herschel-bulkley-determinant-fit/blob/main/web/herschel_bulkley_fit.html
- **Offline:** download [`web/herschel_bulkley_fit.html`](web/herschel_bulkley_fit.html) and open it in any browser.

Generate synthetic data, or upload/paste a CSV of shear rate (1/s) and shear stress (Pa); the app returns σ₀, K and n with confidence intervals, the flow curve, the determinant F̄(n) with its root, residuals, goodness-of-fit statistics and a comparison with the Newtonian, power-law and Bingham models. Results can be downloaded as CSV/JSON.

## Contents

- `hb_mullineux/core.py` — the method: scaled determinant `F̄(n)` (Eqs. 5–6), root search, linear solve for `y0`, `K`, approximate standard errors, the yield-estimate method of Section 2, and a Levenberg–Marquardt cross-check.
- `hb_mullineux/data.py` — synthetic flow-curve generator and the paper's "Chlorine" data set (Table 1).
- `hb_mullineux/figures.py`, `reproduce_paper.py` — reproduce Figures 1–5 of the paper plus a synthetic demonstration (fit, determinant, residuals, model comparison) into `outputs/`.
- `hb_mullineux/cli.py` — fit a CSV file from the command line.
- `web/herschel_bulkley_fit.html` — the browser app described above (no server, no build step, no external libraries).
- `data/` — example CSV files: a synthetic flow curve; the chlorine data in the fitted form `(x = e^{-t}, y)` (use `--n-max 1.4`, as in the paper) and, for reference, as the original time series `(t, y)` — the latter is not itself Herschel–Bulkley input.
- `hb_mullineux/metrics.py` — goodness-of-fit statistics, confidence intervals, and the nested-model comparison (Newtonian / power law / Bingham vs Herschel–Bulkley with F-tests).
- `tests/` — pytest suite (15 tests).

## Quick start

```bash
pip install -r requirements.txt
python reproduce_paper.py --output outputs            # Figures 1-5 + demo, metrics.json
python -m hb_mullineux.cli data/synthetic_herschel_bulkley.csv --plot fit.png --json fit.json
python -m pytest -q
```

In Python:

```python
from hb_mullineux import fit_herschel_bulkley, synthetic_data

x, y = synthetic_data(y0=5, K=4, n=0.35, noise=0.03, seed=1)   # shear rate (1/s), stress (Pa)
fit = fit_herschel_bulkley(x, y)
print(fit.summary())          # y0, K, n, S, E, R^2, std. errors, 95% CIs, AICc/BIC, Durbin-Watson
print(fit.gof.as_dict())      # all goodness-of-fit statistics
from hb_mullineux import compare_nested_models
print(compare_nested_models(x, y, fit, {"y0": fit.y0, "K": fit.K, "n": fit.n}).table())
print(fit(np.array([10.0])))  # evaluate the fitted curve
```

The CSV reader accepts an optional header row; comma, semicolon, tab or space delimiters; and decimal commas. Columns are picked by header name (`--rate-col`, `--stress-col`) or default to the first two numeric columns.

## What is reproduced

| Paper | Here | Result |
|---|---|---|
| Fig. 1 | `figure_01_hb_curve.png` | the illustrative curve `y0=5`, `K=4`, `n=0.35` |
| Section 2, Fig. 2 | `figure_02_yield_estimate_sensitivity.png` | `K`, `10n`, `10E` against the error `Δy0` in the guessed yield stress, for data sets starting at `x = 5, 10, 20, 40, 80`; a 1 Pa error gives ≈12–14 % error in `K` and ≈5–6 % in `n`, as stated in the paper |
| Figs. 3–4 | `figure_03_determinant_wide.png`, `figure_04_determinant_zoom.png` | `F̄(n)` for the 80-point data set: zero at `n=0`, positive peak ≈10 near `n=0.22`, root at `n=0.3500`, minimum ≈−210 near `n=1.5` |
| Section 7, Fig. 5 | `figure_05_chlorine_example.png` | chlorine data: `a = 39.09`, `b = 0.828`, `n = 0.159` (paper: 39.09, 0.828, 0.159) |

On exact data the method recovers `(5, 4, 0.35)` to 1e-9. On noisy synthetic data the solution coincides with SciPy's Levenberg–Marquardt to 1e-6 relative — as it must, because the consistency condition is the stationarity condition of the residual sum of squares.

## Goodness of fit

Both implementations report the same set of statistics for the fitted curve (`fit.gof` in Python, the "Goodness of fit" block in the app, and both JSON exports). With $m$ points, $p = 3$ parameters and residuals $r_i = \sigma_{\text{fit}}(\dot\gamma_i) - \sigma_i$:

| Statistic | Definition | How to read it |
|---|---|---|
| $S$ | $\sum r_i^2$ | the quantity the fit minimises (paper Eq. 2) |
| $E$ (RMSE) | $\sqrt{S/m}$ | the paper's error measure, in Pa |
| $s$ | $\sqrt{S/(m-p)}$ | unbiased estimate of the measurement noise; residuals should mostly lie within $\pm 2s$ |
| MAE, max $\lvert r\rvert$ | mean and largest absolute residual | outliers show up in the maximum |
| MAPE | $\tfrac{100}{m}\sum \lvert r_i/\sigma_i\rvert$ | relative error, comparable across data sets of different stress level |
| $R^2$, adjusted $R^2$ | $1 - S/\sum(\sigma_i-\bar\sigma)^2$, adjusted for $p$ | close to 1 for any smooth monotone curve — necessary, not sufficient |
| AIC, AICc, BIC | $m\ln(S/m) + 2p$, small-sample correction, $m\ln(S/m) + p\ln m$ | only differences between models on the *same* data mean anything; a difference above ~10 is decisive |
| Durbin–Watson | $\sum(r_i-r_{i-1})^2 / S$ | 2 for independent residuals; well below 2 means the residuals run in same-sign stretches, i.e. the model shape is wrong even if $R^2$ is high |
| standard errors, 95 % CI | from $s^2 (J^\top J)^{-1}$ and the $t_{m-p}$ quantile | linearised, so approximate; the app also shows the parameter correlations, which are strongly negative between $K$ and $n$ (they trade off) |

The residual plot (`demo_goodness_of_fit.png`, and the app) is the most informative single check: structure in the residuals — a bow, a trend at low shear rates — indicates wall slip, thixotropy, or a range where the model does not apply, none of which the summary numbers alone reveal.

### Is the full model needed?

The Herschel–Bulkley model contains the Newtonian ($\sigma = \eta\dot\gamma$), power-law ($\sigma_0 = 0$) and Bingham ($n = 1$) models as special cases. `compare_nested_models` (Python) and the "Simpler models" table (app) fit all three by least squares on the original scale and test each against the full fit with the extra-sum-of-squares F-test,

$$F = \frac{(S_{\text{reduced}} - S_{\text{full}})/(p_{\text{full}} - p_{\text{reduced}})}{S_{\text{full}}/(m - p_{\text{full}})},$$

together with AICc and BIC. A p-value above 0.05 for the Bingham comparison means the data do not support a flow index different from 1; above 0.05 for the power law means the yield stress is not resolved by the data — the usual situation when the lowest measured shear rate is too high. `tests/test_metrics.py` checks that the comparison flags Bingham data as Bingham and Herschel–Bulkley data as Herschel–Bulkley.

The CLI prints the full table:

```
python -m hb_mullineux.cli data/synthetic_herschel_bulkley.csv
```

## Notation

The shear rate is written $\dot\gamma$ throughout. In the figures this is matplotlib mathtext (`$\dot{\gamma}$`); in the app the headline equation is MathML and the SVG axis labels draw the dot explicitly, because the Unicode combining dot (`γ̇`, U+03B3 U+0307) does not stack reliably in most fonts. CSV headers and JSON keys use ASCII names (`shear_rate_1_s`, `shear_stress_Pa`).

## Implementation notes

- The root search scans `F̄(n)` on a grid over `[0.01, n_max]` (default `n_max = 5`), refines every sign change by Brent's method, and — because noisy data can in principle produce more than one root — evaluates `y0`, `K` and the residual `S` at each and keeps the least-squares minimiser. The trivial root at `n = 0` (two identical columns) is excluded by starting the scan above zero.
- Negative `K` is allowed, so decaying data such as the chlorine example (`y = a − ab x^n`) fit directly; `F(n)` is then reflected in the axis, as in the paper's Fig. 5.
- Standard errors are the usual linearised `s² (JᵀJ)⁻¹` at the solution; they are approximate.
- The browser app implements the same arithmetic in JavaScript (determinant, bisection, 2×2 solve, standard errors, yield-estimate table, Levenberg–Marquardt check). Its results agree with the Python code to full double precision on the reference cases.

## License

MIT (see `LICENSE`). The reproduced paper is © Elsevier; the "Chlorine" data are the Bates & Watts data set as tabulated in the paper.
