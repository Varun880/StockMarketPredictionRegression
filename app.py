import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
regressor = joblib.load("stock_price_model.pkl")
df = pd.read_csv("stock_data.csv",skiprows=2, index_col="Date", parse_dates=True)

st.title("📈 Stock Price Predictor with Linear Regression")

# User inputs for new features
close_price = st.number_input("Enter today's closing price:")
ma_5 = st.number_input("Enter 5-day moving average:")
ma_10 = st.number_input("Enter 10-day moving average:")
ma_20 = st.number_input("Enter 20-day moving average:")
daily_return = st.number_input("Enter today's daily return (%):")
price_range = st.number_input("Enter today's high-low price difference:")
volume_change = st.number_input("Enter today's volume change (%):")

option = st.selectbox("Select Price Type:", ["Close", "Open", "High", "Low"])

# Plot the selected price data
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df[option], label=f"{option} Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title(f"{option} Price Over Time")
ax.legend()
ax.grid()

# Display in Streamlit
st.pyplot(fig)

# Predict button
if st.button("Predict Next Price"):
    features = np.array([[close_price, ma_5, ma_10, ma_20, daily_return, price_range, volume_change]])
    predicted_price = regressor.predict(features)
    st.success(f"Predicted Stock Price for Next Day: ${predicted_price[0]:.2f}")
