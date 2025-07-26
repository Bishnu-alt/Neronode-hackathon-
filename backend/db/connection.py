import mysql.connector
import yaml
import os

def get_db_connection():
    # ✅ Use absolute path to avoid FileNotFoundError
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    conn = mysql.connector.connect(
        host=cfg["db"]["host"],
        user=cfg["db"]["user"],
        password=cfg["db"]["password"],
        database=cfg["db"]["name"]
    )
    return conn
