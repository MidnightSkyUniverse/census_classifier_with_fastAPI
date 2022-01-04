from fastapi.testclient import TestClient
from main import app
import pytest

# Since I introduced model load on startup in main.py, there was an issue with testing
# models as they were not loading. This structure allows to pass tests with async model load
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to FastAPI interface to Rnadom Forest Classifier of census data"}


def test_predict(client, json_sample):
    response = client.post("/predict", json=json_sample)
    assert response.status_code == 200
    assert response.json()['result'] == 1


def test_predict_2(client, json_sample_2):
    response = client.post("/predict", json=json_sample_2)
    assert response.status_code == 200
    assert response.json()['result'] == 0

def test_predict_error_422(client, json_sample_with_error):
    response = client.post("/predict", json=json_sample_with_error)
    assert response.status_code == 422
