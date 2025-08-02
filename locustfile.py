from locust import HttpUser, task

class APIUser(HttpUser):
    @task
    def predict_text(self):
        self.client.post("/predict", json={"text": "I need mentorship for my startup"})
