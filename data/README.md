# Local dataset

Place the private daily market file at `data/price.csv`. The file is ignored by
Git and is never downloaded by this project.

Canonical columns are `date`, `code`, `open`, `high`, `low`, `close`, `volume`,
and optional `amount`, `vwap`, `adj_factor`, `industry`, and `market_cap`.
Common Chinese and vendor aliases are mapped automatically. `industry` and
`market_cap` are optional risk-exposure inputs; if absent, the run metadata
explicitly records that the corresponding penalty was not applied.

The preparation stage writes `data/daily_price.pkl` locally.

