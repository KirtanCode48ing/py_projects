import yfinance as yf
import numpy as np
import pandas as pd

tickers = ["KO", "PEP"]
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")

close = data["Close"]

x = close["KO"]
y = close["PEP"]

print(x.head())
print()
print(y.head())
print()
print("Correlation is: ", x.corr(y))

beta ,alpha = np.polyfit(y, x, 1)

print("\nBeta: ", beta)
print("Alpha(intercept): ", alpha)

spread = x - (beta*y + alpha)

mean = spread.mean()
std = spread.std()
zscore = (spread-mean)/std

upper = 2
lower = -2

signals = np.where(zscore > upper, -1,   
          np.where(zscore < lower,  1,  
                    0))                 

signals = pd.Series(signals, index=spread.index)

returns_x = x.pct_change()
returns_y = y.pct_change()

pnl = signals.shift(1)*(returns_x - returns_y)

cum_pnl = (1 + pnl.fillna(0)).cumprod()
print("\nfinal returns: ", cum_pnl.iloc[-1])