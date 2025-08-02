from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgres://avnadmin:AVNS__HsEEMjwSRpNs_J7wdf@pg-78bfe1b-uruti-app.d.aivencloud.com:24935/uruti_db?sslmode=require')
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
