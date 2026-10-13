# VaR estimation for a stock portfolio (MF731 HW 5, sheet `05_var_est_hw1_pub`)

One-day 95% VaR of a $1M cap-weighted MSFT/AAPL/GOOGL portfolio, empirical vs
normal-EWMA, full/linear/quadratic losses. Empirical full 21,043; normal-EWMA full
16,610 (linear exact 16,869).

## Run

```bash
pip install -r requirements.txt
python src/main.py        # writes figures/ and generated/
latexmk -pdf main.tex
```

Prices are Yahoo Finance closes downloaded 2026-10-12 and frozen in `data/`.

## Layout

```
src/common.py      data loading, empirical VaR / ES estimators
src/estimation.py  weights, loss operators, EWMA, VaR estimates
src/main.py        runs it, figure, generated/values.tex
PSEUDOCODE.md      pseudocode
DEFENSE.md         notes for questions (not for submission)
```
