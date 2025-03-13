import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

matplotlib.use('TkAgg')

df = pd.read_csv("processed_stock_data.csv")
features = ["Close", "MA_5", "MA_10", "MA_20", "Daily_Return", "Price_Range", "Volume_Change"]

# Convert Date column to datetime and set as an index
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df.set_index("Date", inplace=True)

X = df[features]
y = df["Close"].shift(-1)

# Dropping last row as it has no next day's price
X = X[:-1]
y = y[:-1]

# Splitting data in Train, Test Data
# shuffle = false so data remains in chronological order because they are time-dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Training the model

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# making predictions
y_pred = regressor.predict(X_test)

# Evaluating the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Visualizing Results
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(y_test):], y_test, label="Actual Prices", color='blue')
plt.plot(df.index[-len(y_test):], y_pred, label="Predicted Prices", linestyle='dashed', color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction vs Actual")
plt.legend()
plt.show()

# Predict Single value based on last data
latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = regressor.predict(latest_data)

print(f"Predicted Stock Price for Next Day: {predicted_price[0]:.2f}")

joblib.dump(regressor, "stock_price_model.pkl")
# loaded_model = joblib.load("stock_price_model.pkl")
