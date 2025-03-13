import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

df = pd.read_csv('stock_data.csv', skiprows=2, index_col='Date', parse_dates=True)
print(df.head())
print(df.info())

# MA -> Moving Averages
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()

# daily return percentage
df["Daily_Return"] = df["Close"].pct_change()

# high-low price difference
df["Price_Range"] = df["High"] - df["Low"]

# volume change percentage
df["Volume_Change"] = df["Volume"].pct_change()  # pct_change() calculates the percentage change from one row to the next.

df.dropna(inplace=True)
df.to_csv("processed_stock_data.csv")
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.title('Stock Closing Price Over Time')
plt.legend()
plt.grid()
plt.savefig("plot.png")  # Saves plot as an image
plt.show()
