# Computing Value at Risk (MF731 HW 4)

Closed-form VaR_alpha for three loss distributions:

1. Double-sided exponential with threshold l0 (alpha >= F(l0)):
   `VaR = l0 + (1/b) ln((1 - F(l0)) / (1 - alpha))`
2. Binomial(n, p): smallest k with F(k) >= alpha. For n = 6, p = 1/2, alpha = 0.9: **VaR = 5**.
3. L = Y/Z, Y ~ Exp(lambda), Z ~ Exp(theta) independent:
   `VaR = (theta/lambda) * alpha / (1 - alpha)`

## Run

```bash
pip install -r requirements.txt
python src/var_check.py
latexmk -pdf main.tex
```

`var_check.py` compares each closed form with direct CDF inversion and a
2,000,000-draw Monte Carlo quantile, writes `generated/checks.tex` and
`figures/var_cdfs.png`.
