from app import db

class UserRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(500), nullable=False)
    predicted_label = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<UserRequest {self.id}>'
