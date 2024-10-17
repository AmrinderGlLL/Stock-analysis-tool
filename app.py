from flask import Flask, render_template, request
from app.data import fetch_stock_data
from app.model import prepare_data, train_model, insert_predictions
from datetime import timedelta
import mysql.connector
from app.config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def connect_to_db():
    conn = mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME
    )
    return conn

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        num_days = int(request.form['num_days'])

        # Fetch stock data
        stock_data = fetch_stock_data(ticker)

        # Prepare the data and train the model
        (X_train, X_test, y_train, y_test), df = prepare_data(stock_data)
        model = train_model(X_train, y_train)

        # Make predictions
        last_date = df['date'].iloc[-1]
        next_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

        # Prepare features for future predictions
        last_features = df[['open', 'high', 'low', 'close', 'volume']].tail(1)
        future_predictions = []

        for _ in range(num_days):
            pred = model.predict(last_features)[0]  # Predict the next day
            future_predictions.append(pred)

            # Update last_features for the next day's prediction
            last_features = last_features.copy()
            last_features.iloc[0, 3] = pred  # Assuming you want to update 'close' with predicted value

        # Store predictions in DB
        db_connection = connect_to_db()
        insert_predictions(db_connection, future_predictions, next_dates)
        db_connection.close()

        # Create results for rendering
        results = list(zip(next_dates, future_predictions))

        # Create a plot
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['close'], label='Historical Prices', color='blue')
        plt.plot(next_dates, future_predictions, label='Predicted Prices', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Stock Price Predictions for {ticker}')
        plt.legend()

        # Save the plot to a BytesIO object and encode it
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Render the results template
        return render_template('results.html', results=results, plot_url=plot_url)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
