import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

tickers = ["BTC-USD", "ETH-USD", "DOGE-USD"]
prices = pd.DataFrame({t: yf.download(t)['Close'] for t in tickers})
returns = prices.pct_change().dropna()

alpha = 0.05
VaR = returns.quantile(alpha) #calculate VaR
print("Individual Asset VaR:", VaR)

#Calculate conditional VaR using quantile regression
CVaR = {}
for t1 in tickers:
    for t2 in tickers:
        if t1 != t2:
            X = sm.add_constant(returns[t2])
            y = returns[t1]
            model = sm.QuantReg(y, X)
            res = model.fit(q=alpha)
            CVaR[(t1, t2)] = res.predict([1, VaR[t2]])[0]

CVaR_df = pd.DataFrame(index=tickers, columns=tickers)
for (t1, t2), val in CVaR.items():
    CVaR_df.loc[t1, t2] = val
print("Conditional VaR:", CVaR_df)

#Simulate a simple internal shock to BTC (-30%)
shock = -0.3
returns_shock = returns.copy()
returns_shock['BTC-USD'] += shock

#CVaR under shock
CVaR_shock = {}
for t1 in tickers:
    for t2 in tickers:
        if t1 != t2:
            X = sm.add_constant(returns_shock[t2])
            y = returns_shock[t1]
            model = sm.QuantReg(y, X)
            res = model.fit(q=alpha)
            CVaR_shock[(t1, t2)] = res.predict([1, VaR[t2]])[0]
