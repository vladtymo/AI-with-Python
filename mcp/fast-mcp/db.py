import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

def get_connection():

    connection_string = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={os.getenv("DB_SERVER")},{os.getenv("DB_PORT")};
    DATABASE={os.getenv("DB_DATABASE")};
    UID={os.getenv("DB_USER")};
    PWD={os.getenv("DB_PASSWORD")};
    TrustServerCertificate=yes;
    """

    return pyodbc.connect(connection_string)