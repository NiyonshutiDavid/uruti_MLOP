from locust import HttpUser, task

class APIUser(HttpUser):
    host = "http://localhost:8080"

    @task
    def predict_text(self):
        self.client.post("/predict", json={"text": "I need mentorship for my startup"})
