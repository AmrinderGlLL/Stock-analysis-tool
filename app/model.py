import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import joblib
from app.data import fetch_stock_data
from app.config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
import mysql.connector

def connect_to_db():
    conn = mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME
    )
    return conn

def prepare_data(stock_data):
    df = pd.DataFrame(stock_data['Time Series (Daily)']).T
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    }).astype(float)

    df['date'] = pd.to_datetime(df.index)
    df = df.sort_values('date')

    df['target'] = df['close'].shift(-1)
    df = df[:-1]  # Drop the last row with NaN target

    X = df[['open', 'high', 'low', 'close', 'volume']]
    y = df['target']

    return train_test_split(X, y, test_size=0.2, random_state=42), df

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def insert_predictions(db_connection, predictions, dates):
    cursor = db_connection.cursor()
    insert_query = """
    INSERT INTO stock_predictions (prediction_date, predicted_price)
    VALUES (%s, %s)
    """
    
    for date, price in zip(dates, predictions):
        cursor.execute(insert_query, (date, price))
    
    db_connection.commit()
    cursor.close()

def plot_results(df, y_test, predictions, next_dates, future_predictions):
    plt.figure(figsize=(14, 7))

    # Plot actual vs predicted prices
    plt.plot(df['date'].iloc[-len(y_test):], y_test, label='Actual Prices', color='blue', marker='o')
    plt.plot(df['date'].iloc[-len(predictions):], predictions, label='Predicted Prices', color='orange', marker='x')

    # Plot future predictions
    future_dates = pd.date_range(start=df['date'].iloc[-1] + pd.Timedelta(days=1), periods=2).date
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', marker='s')

    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
