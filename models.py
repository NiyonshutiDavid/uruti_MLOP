from app import db

class UserRequest(db.Model):
    __tablename__ = 'user_request' # Explicitly name the table

    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(500), nullable=False)
    predicted_label = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<UserRequest {self.id}>'

class ModelLog(db.Model):
    __tablename__ = 'model_log' # Explicitly name the table

    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)
    event_details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<ModelLog {self.id} - {self.event_type}>'
