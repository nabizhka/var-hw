# 10-day VaR of a delta-hedged short put (MF731 HW 5, sheet `05_var_10day_hw_pub`)

100 short puts, delta hedged daily. One-day VaR 29.94, sqrt(10) rule 94.68, 10-day
simulation 89.52. Put-call parity check reproduces the lecture's call numbers.

## Run

```bash
pip install -r requirements.txt
python src/main.py        # writes figures/ and generated/
latexmk -pdf main.tex
```

## Layout

```
src/common.py         empirical VaR / ES estimators
src/black_scholes.py  put/call price and delta
src/tenday.py         1-day and 10-day simulation, parity and MC-error checks
src/main.py           runs it, figure, generated/values.tex
PSEUDOCODE.md      pseudocode
DEFENSE.md         notes for questions (not for submission)
```
