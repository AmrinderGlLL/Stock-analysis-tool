# app/__init__.py

# Importing necessary functions and constants from submodules
from .data import fetch_stock_data
from .model import prepare_data, train_model, insert_predictions, plot_results
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME

# Optionally, you could define the version of your package
__version__ = "1.0.0"

# Optional: Define a brief description of the package
__description__ = "A stock analysis tool for fetching data and making predictions using machine learning."
