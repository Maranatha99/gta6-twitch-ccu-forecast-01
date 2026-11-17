"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for each of hours 2..24 of a big game's Twitch launch day, five
quantiles of the category's concurrent viewers, from the hour-1 print and the
launch-day curves of past big launches carried in the case ("comparables").

Method (fitted from the case's own comparables at run time, nothing hard-coded
except the four tuning constants below):

  1. Per hour h, an OLS fit of log(viewers_h) on log(hour-1 print) over all
     comparable launches. The slope is floored at 0.4: the raw fit's evening
     slope collapses toward 0 because a few comparables had a near-dead
     hour 1 (a single global unlock late in the day), which would absurdly
     predict an evening audience independent of launch size. The floor keeps
     the forecast sane when the launch is bigger than every comparable.
  2. A k-nearest-neighbours curve: the 15 comparables whose hour-1 print is
     closest in log space; the median of their log-ratios log(v_h / v_1) is
     the local median curve, and their empirical quantiles give the spread.
  3. The median is a 70/30 blend (in log space) of the OLS prediction and the
     kNN median; the quantiles are the kNN empirical quantiles re-centred on
     that blended median and shrunk by 0.6 (the raw neighbour spread includes
     between-game heterogeneity that overstates the conditional spread).

If too few comparables are usable, the program falls back to the house
persistence baseline (starter/main.py's formula) for the affected hours.

Deterministic: no randomness, no clock, no network; sorted iteration only.
Pure standard library. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA_BASELINE = 0.15  # persistence fallback spread, as in the starter

# Tuning constants (chosen on the four practice cases; see README).
# Retuned 2026-11-17 after a 3x3x3x3 grid search (81 combos) over the same
# four cases: mean log-scale pinball 0.1685 -> 0.1609. The grid's signal was
# consistent, not a single outlier: a higher OLS weight, a smaller shrink,
# and a lower slope floor each helped on their own (0.1665 / 0.1665 /
# 0.1683), and together they gave the best combo. Per case: p00 0.195 ->
# 0.166, p01 0.165 -> 0.173 (slightly worse), p02 0.106 -> 0.106, p03
# 0.208 -> 0.199. k=15 stayed best; k=12/k=20 were both worse alone.
W_OLS = 0.7        # weight of the OLS median in the log-space blend
KNN_K = 15         # neighbours in log hour-1 print
SHRINK = 0.6       # shrink of the neighbours' empirical spread
SLOPE_FLOOR = 0.4  # minimum elasticity of an hour's viewers to hour 1
MIN_CURVES = 5     # fewer usable comparables -> persistence fallback


def parse_ts(s):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantile(sorted_xs, p):
    """Linear-interpolation quantile of a sorted list (p in [0, 1])."""
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    idx = p * (n - 1)
    lo = int(idx)
    frac = idx - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[min(lo + 1, n - 1)] * frac


def comparable_curves(case):
    """Launch-day viewer curves of the case's comparables.

    Returns a list of (hour-1 viewers, {hour: viewers for hours 1..24}).
    The Grand Theft Auto VI pre-release category is not a launch and is
    skipped; comparables with no usable hour-1 print are skipped.
    """
    curves = []
    for comp in case.get("comparables") or []:
        if "grand theft auto vi" in comp.get("name", "").lower():
            continue
        try:
            day_start = parse_ts(comp["day_start"])
            hourly = {}
            for row in comp.get("hourly") or []:
                v = row.get("viewers")
                if v is not None and v > 0:
                    hourly[parse_ts(row["from"])] = v
        except (KeyError, TypeError, ValueError):
            continue
        vals = {}
        for h in range(1, 25):
            v = hourly.get(day_start + timedelta(hours=h - 1))
            if v is not None:
                vals[h] = v
        if vals.get(1) is None or vals[1] <= 0:
            continue
        curves.append((vals[1], vals))
    return curves


def ols_fit(curves):
    """Per hour: (intercept, slope floored at SLOPE_FLOOR, residual sigma, n).

    Fits log(viewers_h) = a + b * log(hour-1 print) over the comparables
    that have an hour-h value.
    """
    fit = {}
    for h in range(2, 25):
        xs, ys = [], []
        for p1, vals in curves:
            if h in vals:
                xs.append(math.log(p1))
                ys.append(math.log(vals[h]))
        n = len(xs)
        if n < 3:
            continue
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx else 0.0
        if b < SLOPE_FLOOR:
            b = SLOPE_FLOOR
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sig = (sum(r * r for r in resid) / max(1, n - 2)) ** 0.5
        fit[h] = (a, b, sig, n)
    return fit


def persistence_quantiles(p1, h):
    """House baseline rows (starter formula), the fallback."""
    return [round2(p1 * math.exp(z * SIGMA_BASELINE * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    curves = comparable_curves(case)
    fit = ols_fit(curves) if len(curves) >= MIN_CURVES else {}

    # The k nearest comparables in log hour-1 print.
    near = sorted(curves, key=lambda c: abs(math.log(c[0]) - log_p1))[:KNN_K]

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if h not in fit or len(near) < 3:
            q = persistence_quantiles(p1, h)
        else:
            a, b, _sig, _n = fit[h]
            mu_ols = a + b * log_p1
            ratios = sorted(
                math.log(vals[h] / vals[1]) for _p1, vals in near if h in vals
            )
            if len(ratios) < 3:
                q = persistence_quantiles(p1, h)
            else:
                mu_knn = log_p1 + quantile(ratios, 0.5)
                mu = W_OLS * mu_ols + (1 - W_OLS) * mu_knn
                med = quantile(ratios, 0.5)
                q = [
                    round2(
                        math.exp(mu + (quantile(ratios, tau) - med) * SHRINK)
                    )
                    for tau in LEVELS
                ]
        # Enforce the output contract: finite, > 0, <= 1e9, non-decreasing.
        clean = []
        prev = 0.01
        for v in q:
            if not math.isfinite(v) or v <= 0:
                v = prev
            v = min(v, 1e9)
            v = max(v, prev)
            clean.append(v)
            prev = v
        hours.append({"hour": h, "q": clean})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
