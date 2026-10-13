# Backtesting VaR on the S&P 500 (MF731 HW 5, sheet `05_var_backtest_hw_pub`)

Rolling one-day 95% VaR for a constant $1 S&P 500 position, M = 1010, J = 1504 test
days. Exceedances: empirical 55, EWMA 94, 95% CI (58.63, 91.77). Both rejected, in
opposite directions.

## Run

```bash
pip install -r requirements.txt
python src/main.py        # writes figures/ and generated/
latexmk -pdf main.tex
```

Prices are Yahoo Finance closes downloaded 2026-10-12 and frozen in `data/`.

## Layout

```
src/common.py    data loading, empirical VaR estimator
src/backtest.py  rolling empirical + EWMA VaR, exceedances, CI
src/main.py      runs it, figure, tables, generated/values.tex
PSEUDOCODE.md      pseudocode
DEFENSE.md         notes for questions (not for submission)
```
