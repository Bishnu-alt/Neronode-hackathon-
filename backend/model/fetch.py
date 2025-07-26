import pickle
import os
from backend.db.connection import get_db_connection


import pickle
from backend.db.connection import get_db_connection

def fetch_global_model(model_id: int = 1, version: int = None):
    """
    Fetch the latest global model from central_updates table.
    If version is None, fetch the latest version by created_at.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if version:
        query = """
            SELECT model_blob 
            FROM central_updates 
            WHERE model_id = %s AND version = %s
        """
        cursor.execute(query, (model_id, version))
    else:
        query = """
            SELECT model_blob 
            FROM central_updates 
            WHERE model_id = %s 
            ORDER BY version DESC 
            LIMIT 1
        """
        cursor.execute(query, (model_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError("Global model not found in central_updates")

    model_blob = row[0]
    if isinstance(model_blob, str):
        model_blob = model_blob.encode('latin1')  # Ensure binary format
    model = pickle.loads(model_blob)
    return model

def save_model_from_db(model_id: int = 1, version: int = None, save_path: str = "backend/model/global_model.pkl"):
    model = fetch_global_model(model_id, version)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)