from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {	"message":	"Welcome to FastAPI interface to Rnadom Forest Classifier of census data"}


def test_predict(json_sample):
    response = client.post("/predict",json=json_sample)
    assert response.status_code == 200
    assert response.json()['result'] == 1

def test_predict_2(json_sample_2):
    response = client.post("/predict",json=json_sample_2)
    assert response.status_code == 200
    assert response.json()['result'] == 0

