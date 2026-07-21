# Daily replication methodology

## Point-in-time boundary

At date `t`, an expression may use only same-day or earlier OHLCV/VWAP. The
forward return from `t` to `t+5` is created only inside reward, AlphaEval and
LightGBM modules. RQAlphaPlus trades a score only when `signal_date < trade_date`.

Rolling functions are grouped by security after stable sorting. Cross-sectional
rank/z-score/demean operations are grouped by date. Missing prices are only
forward-filled; no future observation is used to repair history.

## Safe operator semantics

- `log(x) = log(abs(x) + 1e-12)`
- `sqrt(x) = sqrt(abs(x))`
- `div(x, y)` returns missing when `abs(y) <= 1e-12`
- windows require a complete lookback except delay/delta, which naturally return
  missing until the lag exists
- non-finite results remain missing and are excluded from evaluation

## Reward

Daily Spearman IC is calculated only on dates meeting the minimum cross-section.
Top-decile return is measured relative to the equal-weight cross-sectional mean.

```text
reward = abs(mean RankIC)
       * max(0.05, 1 + clip(LongIR, -0.95, 5))
       * exp(-risk_aversion * (industry_exposure + size_exposure))
```

Industry exposure is the daily one-hot regression R-squared. Size exposure is
absolute daily Spearman correlation with log market cap. Each term is exactly
zero, and its applied flag false, if the real input field is absent.

## AlphaEval mini version

- PPS: absolute RankIC and ICIR.
- Temporal stability: rolling RankIC dispersion and sign consistency.
- Perturbation robustness: IC retention after seeded cross-sectional noise.
- Financial logic: deterministic penalty for excessive length/depth.
- RRE: adjacent-date rank correlation and average absolute rank change.
- Diversity: quality-weighted DPP greedy MAP selection and correlation penalty.

## Validation caveats

Mining, AlphaEval, hyperparameter selection and performance reporting should use
separate time periods in a production study. The code purges overlapping labels
from each LightGBM fold, but the user must reserve a final untouched evaluation
period. The report's proprietary data universe and Barra data are not inferred.

