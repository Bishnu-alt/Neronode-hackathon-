import mysql.connector

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="192.168.91.196",
            user="clientUsers",
            password="ronaldo7",
            database="fl_database"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        raise