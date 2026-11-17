# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Maranatha99/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Maranatha99/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), as five quantiles an hour, from the first hour's viewer count and the launch-day curves of past big game launches. Pure Python 3.13 standard library, deterministic, no network.

## Method

The program reads the case's `comparables` (past launches' hourly Twitch curves) at run time and fits, per forecast hour:

1. **OLS on log scale** — `log(viewers_h)` on `log(hour-1 print)`. The evening slope is floored at 0.5: the raw fit collapses toward 0 because a few comparables had a near-dead hour 1 (a single global unlock late in the day), which would absurdly predict an evening audience independent of launch size.
2. **k-nearest-neighbour curve** — the 15 comparables closest in log hour-1 print; the median of their log-ratios `log(v_h / v_1)` is the local median curve and their empirical quantiles give the spread.
3. **Blend** — the median is a 60/40 blend (in log space) of the OLS prediction and the kNN median; quantiles are the neighbours' empirical quantiles re-centred on that median and shrunk by 0.7.

If too few comparables are usable, the program falls back to the house persistence baseline for the affected hours. See `main.py` for the full description.

## Practice results

Pinball loss on the log scale (lower is better), scored against the actual hourly counts of the four practice launches (SullyGnome hourly averages):

| case | entry | house baseline |
| --- | --- | --- |
| p00 Marvel's Wolverine | 0.195 | 0.744 |
| p01 Red Dead Redemption 2 | 0.165 | 0.287 |
| p02 God of War Ragnarök | 0.106 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.208 | 0.303 |
| **mean** | **0.169** | **0.387** |

The entry beats the house baseline on all four practice cases. Typical miss ~16–38% (baseline: 42–189%). 90% coverage 0.74–1.00 across cases.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=s11m3687&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
