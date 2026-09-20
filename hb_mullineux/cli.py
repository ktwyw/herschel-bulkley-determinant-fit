"""Command-line fitting of a shear-stress / shear-rate CSV file.

Usage::

    python -m hb_mullineux.cli data.csv [--rate-col NAME] [--stress-col NAME]
                                        [--n-max 5] [--plot fit.png] [--json out.json]

The file may have a header row.  Columns are chosen by name if given,
otherwise the first two numeric columns are used as (shear rate, shear stress).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .core import fit_herschel_bulkley, refine_least_squares
from .metrics import compare_nested_models


def read_csv(path: Path, rate_col: str | None = None, stress_col: str | None = None
             ) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("The file is empty.")

    def is_num(s: str) -> bool:
        try:
            float(s.replace(",", "."))
            return True
        except ValueError:
            return False

    # Delimiter: the first of ; tab , space that yields >= 2 numeric columns
    # in every data row (';' and tab first, so decimal commas survive).
    rows: list[list[str]] | None = None
    for delim in (";", "\t", ",", None):
        cand = [[c.strip() for c in (ln.split(delim) if delim else ln.split())] for ln in lines]
        body_c = cand[1:] if len(cand) > 1 else cand
        ncol_c = min(len(r) for r in body_c)
        numeric_cols = [j for j in range(ncol_c) if all(is_num(r[j]) for r in body_c)]
        if len(numeric_cols) >= 2:
            rows = cand
            break
    if rows is None:
        raise ValueError("Could not find two numeric columns (shear rate, shear stress).")

    header = rows[0] if not all(is_num(c) for c in rows[0] if c.strip()) else None
    body = rows[1:] if header is not None else rows
    ncol = max(len(r) for r in body)
    if header is not None and (rate_col or stress_col):
        names = [h.strip().lower() for h in header]
        ir = names.index(rate_col.strip().lower()) if rate_col else None
        is_ = names.index(stress_col.strip().lower()) if stress_col else None
    else:
        ir = is_ = None
    if ir is None or is_ is None:
        # first two columns that are numeric in every row
        numeric = [j for j in range(ncol) if all(j < len(r) and is_num(r[j]) for r in body)]
        if len(numeric) < 2:
            raise ValueError("Could not find two numeric columns (shear rate, shear stress).")
        ir = ir if ir is not None else numeric[0]
        is_ = is_ if is_ is not None else next(j for j in numeric if j != ir)
    x = np.array([float(r[ir].replace(",", ".")) for r in body])
    y = np.array([float(r[is_].replace(",", ".")) for r in body])
    labels = (header[ir], header[is_]) if header is not None else (f"column {ir + 1}", f"column {is_ + 1}")
    return x, y, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit y = y0 + K x^n by the Mullineux (2008) method.")
    parser.add_argument("csv", type=Path, help="CSV with shear rate (1/s) and shear stress (Pa) columns")
    parser.add_argument("--rate-col", help="header name of the shear-rate column")
    parser.add_argument("--stress-col", help="header name of the shear-stress column")
    parser.add_argument("--n-max", type=float, default=5.0, help="upper end of the search interval for n")
    parser.add_argument("--plot", type=Path, help="write a PNG of the data, fit and F(n)")
    parser.add_argument("--json", type=Path, help="write the fitted parameters as JSON")
    args = parser.parse_args(argv)

    x, y, labels = read_csv(args.csv, args.rate_col, args.stress_col)
    print(f"Read {x.size} points from {args.csv} (rate: {labels[0]}, stress: {labels[1]})")
    fit = fit_herschel_bulkley(x, y, n_max=args.n_max)
    print(fit.summary())
    lm = refine_least_squares(x, y, fit.y0, fit.K, fit.n)
    print(f"  Levenberg-Marquardt check from this solution: y0={lm.y0:.6g}, K={lm.K:.6g}, n={lm.n:.6g}")
    comparison = compare_nested_models(x, y, fit, {"y0": fit.y0, "K": fit.K, "n": fit.n})
    print("\nIs the full model needed?  Nested-model comparison on the same data:")
    print(comparison.table())

    if args.json:
        payload = {
            "method": "Mullineux (2008) consistency-condition",
            "y0": fit.y0, "K": fit.K, "n": fit.n, "S": fit.S, "rmse": fit.rmse, "r_squared": fit.r_squared,
            "standard_errors": fit.standard_errors, "m": fit.m,
            "goodness_of_fit": fit.gof.as_dict(),
            "residuals": fit.gof.residuals.tolist(),
            "model_comparison": {
                "models": [{"name": mf.name, "p": mf.p, "S": mf.S, "rmse": mf.rmse, "r_squared": mf.r_squared,
                            "aicc": mf.aicc, "bic": mf.bic, "parameters": mf.parameters} for mf in comparison.models],
                "f_tests": [{"reduced": t.reduced, "full": t.full, "F": t.F, "df": [t.df1, t.df2], "p_value": t.p_value}
                            for t in comparison.tests],
            },
            "roots_considered": [{"n": c.n, "y0": c.y0, "K": c.K, "S": c.S} for c in fit.candidates],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  parameters written to {args.json}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .core import determinant

        fig, (a, c, b) = plt.subplots(1, 3, figsize=(15, 4.5))
        xx = np.geomspace(x.min(), x.max(), 300)
        a.plot(x, y, "o", ms=4, label="data")
        a.plot(xx, fit(xx), color="tab:red", lw=2, label=f"y0={fit.y0:.3g}, K={fit.K:.3g}, n={fit.n:.3g}")
        a.set_xscale("log"); a.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)"); a.set_ylabel(r"Shear stress $\sigma$ (Pa)")
        a.legend(); a.grid(alpha=0.2, which="both")
        g = fit.gof
        c.axhline(0, color="k", lw=0.8); c.plot(x, g.residuals, "o", ms=4)
        c.axhspan(-2 * g.residual_std_error, 2 * g.residual_std_error, color="tab:blue", alpha=0.08)
        c.set_xscale("log"); c.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)"); c.set_ylabel("Residual (Pa)")
        c.set_title(f"s = {g.residual_std_error:.3g}, MAPE = {g.mape:.2f}%, DW = {g.durbin_watson:.2f}", fontsize=10); c.grid(alpha=0.2, which="both")
        nn = np.linspace(0.0, max(2.0, 1.5 * fit.n), 1000)
        b.plot(nn, determinant(nn, x, y), lw=1.8); b.axhline(0, color="k", lw=0.8)
        b.plot([fit.n], [0], "o", color="tab:red"); b.set_xlabel("n"); b.set_ylabel(r"$\bar F(n)$"); b.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(args.plot, dpi=160)
        print(f"  plot written to {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
