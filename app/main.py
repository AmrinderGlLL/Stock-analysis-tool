from app.data import fetch_stock_data
from app.model import prepare_data, train_model, insert_predictions, plot_results
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import pandas as pd
import mysql.connector
from app.config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME

def connect_to_db():
    conn = mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME
    )
    return conn

def main():
    try:
        # Get user input for stock ticker and number of days
        ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
        num_days = int(input("Enter the number of days for predictions (e.g., 2): "))

        # Fetch stock data
        stock_data = fetch_stock_data(ticker)

        # Prepare the data and train the model
        (X_train, X_test, y_train, y_test), df = prepare_data(stock_data)
        model = train_model(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        print(f'Mean Absolute Error: {mae}')

        # Get next dates for predictions
        last_date = df['date'].iloc[-1]
        next_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]
        
        # Prepare the last input features for future predictions
        last_features = df[['open', 'high', 'low', 'close', 'volume']].tail(1).values  # Get the last row
        future_predictions = []

        for _ in range(num_days):
            prediction = model.predict(last_features)
            future_predictions.append(prediction[0])

            # Update the last_features for the next prediction
            last_features[0][3] = prediction[0]  # Update 'close' for the next iteration
            # You may want to also modify 'open', 'high', 'low' accordingly if needed

        # Print predictions
        print("Predicted Prices:")
        for date, price in zip(next_dates, future_predictions):
            print(f"{date.date()}: {price}")

        # Connect to the database and store predictions
        db_connection = connect_to_db()
        insert_predictions(db_connection, future_predictions, next_dates)
        db_connection.close()

        # Ensure to plot using the correct sizes
        plot_results(df, y_test, predictions, next_dates, future_predictions)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
