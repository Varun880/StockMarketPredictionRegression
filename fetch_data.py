import yfinance as yf

ticker = 'GOOGL'
start_date = "2020-01-01"
end_date = "2024-01-01"

stock_data = yf.download(ticker, start=start_date, end=end_date)

stock_data.to_csv("stock_data.csv")
print(stock_data.head())