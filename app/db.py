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
