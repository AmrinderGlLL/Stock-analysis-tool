from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API Configuration
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# MySQL Database Configuration
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = 'stock_analysis'
