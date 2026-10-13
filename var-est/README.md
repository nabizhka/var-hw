# Expected shortfall and subadditivity (MF731 HW 6)

1. N(0, 20^2): VaR.99 = 46.53m, ES.99 = 53.30m.
2. 0/10 losses: VaR.98 0 + 0 < 10 (not subadditive); VaR.99 10 <= 20; ES.98 = 10, 10, 10.2.
3. Loans: VaR.99 loan 2m, portfolio 5.8m > 4m (not subadditive); ES.99 6m and 7.8m.
4. Weak monotonicity proved; the strict claim is false as stated (counterexample
   L2 = U, L1 = U - |U - alpha|/2), true under a bounded gap or F2(VaR) > alpha.
5. Exp: VaR = -theta ln(1 - alpha), ES = VaR + theta. Pareto: VaR = (1 - alpha)^(-xi),
   ES = VaR / (1 - xi) for xi < 1, infinite for xi >= 1.

```bash
pip install -r requirements.txt
python src/es_check.py     # writes generated/checks.tex
latexmk -pdf main.tex
```
