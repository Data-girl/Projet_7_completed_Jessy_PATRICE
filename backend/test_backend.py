import joblib
from fastapi import status
from fastapi.testclient import TestClient

from backend import app

client = TestClient(app=app)
data = joblib.load("./models/df_final.pkl")


def test_database_returns_correct():
    all_clients = data["sk_id_curr"].tolist()
    response = client.get("/database")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"values": all_clients}


def test_predict_correct():
    response = client.post("/predict", params={"numero_client": 100200})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "probabilite": "Prêt accordé, client sûr",
        "prediction": "Il s'agit d'un client sûr",
    }


def test_predict_returns_incorrect():
    response = client.post("/predict", params={"numero_client": "albert972"})
    assert response.status_code == 422
