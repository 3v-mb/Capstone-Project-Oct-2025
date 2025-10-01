import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]
prices = pd.DataFrame({t: yf.download(t)['Adj Close'] for t in tickers})
returns = prices.pct_change().dropna()

alpha = 0.05
VaR = returns.quantile(alpha) #computer VaR
print("Individual Asset VaR (5%):\n", VaR)

#Compute CoVaR (conditional VaR using quantile regression)
coVaR = {}
for t1 in tickers:
    for t2 in tickers:
        if t1 != t2:
            X = sm.add_constant(returns[t2])
            y = returns[t1]
            model = sm.QuantReg(y, X)
            res = model.fit(q=alpha)
            coVaR[(t1, t2)] = res.predict([1, VaR[t2]])[0]

# 4. Display CoVaR table
coVaR_df = pd.DataFrame(index=tickers, columns=tickers)
for (t1, t2), val in coVaR.items():
    coVaR_df.loc[t1, t2] = val
print("\nConditional VaR (CoVaR):\n", coVaR_df)

#Simulate a simple internal shock to BTC (-30%)
shock = -0.3
returns_shock = returns.copy()
returns_shock['BTC-USD'] += shock

# Recompute CoVaR under shock
coVaR_shock = {}
for t1 in tickers:
    for t2 in tickers:
        if t1 != t2:
            X = sm.add_constant(returns_shock[t2])
            y = returns_shock[t1]
            model = sm.QuantReg(y, X)
            res = model.fit(q=alpha)
            coVaR_shock[(t1, t2)] = res.predict([1, VaR[t2]])[0]

print("\nCoVaR under BTC shock:\n", coVaR_shock)
# -------------------- End --------------------
