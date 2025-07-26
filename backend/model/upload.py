import pickle
import mysql.connector
from sklearn.metrics import classification_report
from backend.db.connection import get_db_connection


def upload_model_update(model, y_test, y_pred, accuracy, loss, model_id, client_id, round_num):
    # Serialize the model
    model_blob = pickle.dumps(model)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    macro_f1 = report["macro avg"]["f1-score"]
    recall_minority = report["1.0"]["recall"]
    f1_minority = report["1.0"]["f1-score"]
    f1_majority = report["0.0"]["f1-score"]

    # You can improve this with logic to detect underfit/overfit
    fit_status = "good"

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO client_updates 
    (model_id, client_id, model_blob, accuracy, loss, round_num, macro_f1, recall_minority, f1_minority, f1_majority, fit_status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(insert_query, (
        model_id, client_id, model_blob, accuracy, loss, round_num,
        macro_f1, recall_minority, f1_minority, f1_majority, fit_status
    ))
    conn.commit()
    cursor.close()
    conn.close()

    print(f"[✔] Model update inserted for client {client_id} in round {round_num}")
