import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
stock_symbol = 'AAPL' 
interval = '5min'  

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval={interval}&apikey={API_KEY}'

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("Stock Data Retrieved Successfully!")
    print(data)
else:
    print("Error fetching data:", response.status_code, response.text)
