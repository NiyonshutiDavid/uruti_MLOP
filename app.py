from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://avnadmin:AVNS__HsEEMjwSRpNs_J7wdf@pg-78bfe1b-uruti-app.d.aivencloud.com:24935/defaultdb?sslmode=require')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import models after db is initialized to avoid circular imports
from models import UserRequest, ModelLog
UserRequestModel = UserRequest(db).create_model()
ModelLogModel = ModelLog(db).create_model()

def add_user_request(input_text: str, predicted_label: str):
    new_request = UserRequestModel(input_text=input_text, predicted_label=predicted_label)
    db.session.add(new_request)
    db.session.commit()
    log_model_event("prediction", f"Input: {input_text}, Label: {predicted_label}")

def get_all_user_requests():
    return UserRequestModel.query.all()

def log_model_event(event_type: str, event_details: str = None):
    log_entry = ModelLogModel(event_type=event_type, event_details=event_details)
    db.session.add(log_entry)
    db.session.commit()

def get_user_request_by_id(request_id: int):
    return UserRequestModel.query.get(request_id)

@app.route('/')
def hello_world():
    return 'Hello, Uruti Web App!'

from datetime import datetime
from flask import jsonify, render_template, request
import subprocess
from prometheus_flask_exporter import PrometheusMetrics
import random

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

@app.route('/dashboard')
def dashboard():
    """
    Serve the dashboard page.
    """
    return render_template('dashboard.html')

@app.route('/requests_page')
def requests_page():
    """
    Serve the user requests page.
    """
    return render_template('requests.html')

@app.route('/model_logs', methods=['GET'])
def model_logs():
    """
    Return model logs as JSON.
    """
    logs = ModelLogModel.query.order_by(ModelLogModel.timestamp.desc()).limit(100).all()
    logs_data = [{
        'id': log.id,
        'event_type': log.event_type,
        'event_details': log.event_details,
        'timestamp': log.timestamp.isoformat()
    } for log in logs]
    return jsonify(logs_data)

@app.route('/retrain')
def retrain_page():
    """
    Serve the model retraining page.
    """
    return "<h1>Model Retraining Page - Coming Soon</h1>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
