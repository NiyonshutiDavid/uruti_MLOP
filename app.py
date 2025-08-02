from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://user:password@localhost/uruti_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from models import UserRequest

@app.route('/')
def hello_world():
    return 'Hello, Uruti Web App!'

from datetime import datetime

def add_user_request(input_text: str, predicted_label: str):
    new_request = UserRequest(input_text=input_text, predicted_label=predicted_label)
    db.session.add(new_request)
    db.session.commit()

def get_all_user_requests():
    return UserRequest.query.all()

def get_user_request_by_id(request_id: int):
    return UserRequest.query.get(request_id)
from flask import jsonify

@app.route('/requests', methods=['GET'])
def get_requests():
    """
    Retrieve all user requests from the database.
    """
    requests = get_all_user_requests()
    requests_data = [{
        'id': req.id,
        'input_text': req.input_text,
        'predicted_label': req.predicted_label,
        'timestamp': req.timestamp.isoformat()
    } for req in requests]
    return jsonify(requests_data)

from flask import render_template

@app.route('/dashboard')
def dashboard():
    """
    Serve the dashboard page.
    """
    return render_template('dashboard.html')

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Retrieve model performance metrics.
    """
    # Placeholder metrics - replace with actual calculations later
    performance_metrics = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.90
    }
    return jsonify(performance_metrics)

from flask import render_template

@app.route('/requests_page')
def requests_page():
    """
    Serve the user requests page.
    """
    return render_template('requests.html')

from flask import render_template

@app.route('/retrain')
def retrain_page():
    """
    Serve the model retraining page.
    """
    return render_template('retrain.html')

from flask import jsonify, request

@app.route('/trigger_retraining', methods=['POST'])
def trigger_retraining():
    """
    Endpoint to trigger the model retraining pipeline.
    """
    # Placeholder logic to simulate triggering retraining
    print("Retraining process triggered...")
    # In a real application, you would start a background job here
    # e.g., queue a task, run a script in a new process, etc.

    return jsonify({"status": "success", "message": "Model retraining triggered."})

from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)

# Custom metric to track predicted labels
predicted_labels_total = metrics.counter(
    'predicted_labels_total', 'Total number of predicted labels',
    labels={'label': lambda: request.view_args['label']}
)

import random

@app.route('/predict', methods=['POST'])
@predicted_labels_total
def predict():
    """
    Placeholder for prediction logic.
    """
    # Simulate a prediction
    labels = ["mentorship needed", "investor pitching ready", "rejected need to be worked on"]
    predicted_label = random.choice(labels)
    
    # The decorator will automatically increment the counter
    # with the label returned by the lambda function.
    # To make this work, we need to pass the label in the view_args.
    # A simple way to do this is to have a dynamic route, but for this
    # example, we will just simulate it.
    request.view_args = {'label': predicted_label}
    
    return jsonify({"predicted_label": predicted_label})
import subprocess
from flask import jsonify

@app.route('/trigger_retraining', methods=['POST'])
def trigger_retraining():
    """
    Triggers the retraining script.
    """
    try:
        subprocess.Popen(['python', 'retrain_script.py'])
        return jsonify({"status": "success", "message": "Retraining process started."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
from flask import render_template

@app.route('/retrain')
def retrain_page():
    """
    Serve the model retraining page.
    """
    return render_template('retrain.html')

from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)

# Custom metric to track predicted labels
predicted_labels_total = metrics.counter(
    'predicted_labels_total', 'Total number of predicted labels',
    labels={'label': lambda: request.view_args['label']}
)
import random
from flask import jsonify

@app.route('/predict/<label>')
@predicted_labels_total
def predict(label):
    """
    A dummy predict endpoint that simulates a prediction and increments the
    predicted_labels_total counter.
    """
    # In a real application, you would put your prediction logic here
    return jsonify({"status": "success", "predicted_label": label})
