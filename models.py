from flask_sqlalchemy import SQLAlchemy

class UserRequest:
    def __init__(self, db_instance):
        self.db = db_instance
    
    def create_model(self):
        class UserRequestModel(self.db.Model):
            id = self.db.Column(self.db.Integer, primary_key=True)
            input_text = self.db.Column(self.db.String(500), nullable=False)
            predicted_label = self.db.Column(self.db.String(50), nullable=False)
            timestamp = self.db.Column(self.db.DateTime, server_default=self.db.func.now())

            def __repr__(self):
                return f'<UserRequest {self.id}>'
        
        return UserRequestModel

class ModelLog:
    def __init__(self, db_instance):
        self.db = db_instance

    def create_model(self):
        class ModelLogModel(self.db.Model):
            id = self.db.Column(self.db.Integer, primary_key=True)
            event_type = self.db.Column(self.db.String(50), nullable=False)
            event_details = self.db.Column(self.db.Text, nullable=True)
            timestamp = self.db.Column(self.db.DateTime, server_default=self.db.func.now())

            def __repr__(self):
                return f'<ModelLog {self.id} - {self.event_type}>'

        return ModelLogModel
