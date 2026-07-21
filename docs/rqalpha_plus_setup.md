# RQAlphaPlus setup

RQAlphaPlus is licensed Ricequant software and is intentionally not replaced by
a home-grown simulator. Install `rqalpha_plus` through your authorized
Ricequant distribution, then prepare the local data bundle without enabling
automatic downloads during the experiment.

Run:

```bash
python -m rqalpha_strategy.run_backtest \
  --bundle ~/.rqalpha-plus/bundle \
  --predictions results/lightgbm/prediction_score.csv
```

The strategy uses the most recent signal strictly earlier than the trading day.
The analyzer writes the performance summary, equity curve, positions and trade
history under `results/backtest_report/`.

Official references:

- https://www.ricequant.com/doc/rqalpha-plus/api/config
- https://www.ricequant.com/doc/rqalpha-plus/api/entrypoint
- https://www.ricequant.com/doc/rqalpha-plus/api/order-api

