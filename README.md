# Uruti.Rw MLOps Pipeline 🚀

## 📋 Project Overview

Uruti.Rw is a comprehensive Machine Learning Operations (MLOps) platform designed to classify startup ideas into three distinct categories: **Mentorship Needed**, **Investment Ready**, and **Needs Refinement**. The project implements an end-to-end ML pipeline with audio processing capabilities, allowing users to submit both text and audio inputs for real-time predictions.

### 🎯 Problem Statement
In Rwanda's growing startup ecosystem, there's a need for an intelligent system that can automatically assess startup pitches and ideas, providing valuable insights to entrepreneurs, mentors, and investors about the readiness and potential of business concepts.

### 🔧 Solution Architecture
The platform leverages multiple state-of-the-art NLP models (DistilBERT, RoBERTa, ALBERT) for text classification and OpenAI Whisper for audio transcription, deployed through a scalable FastAPI backend with an HTML/CSS/JavaScript dashboard for monitoring and management.

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Dashboard │    │   Developer     │
│   (End Users)   │────│   (HTML/CSS/JS) │────│   Dashboard     │
│   Flutter/React│    │   Frontend      │    │   (Flask)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   ML Server     │
                    │   (Port 8000)   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Database      │
                    │   (Aiven Cloud) │
                    └─────────────────┘
```

---

## 🚀 Features

### Core ML Capabilities
- **🎤 Audio Processing**: Real-time audio transcription using OpenAI Whisper, the dataset used was downloaded from Mozilla, you can access the dataset (here)[https://drive.google.com/file/d/10WzRvyyau2etuK08T835ndAfGH6y3Axf/view?usp=drive_link]
- **📝 Text Classification**: Multi-model ensemble (DistilBERT, RoBERTa, ALBERT, TensorFlow)
- **🔄 Model Retraining**: Automated retraining pipeline with user data
- **📊 Performance Monitoring**: Real-time metrics and model performance tracking with Weights & Biases integration

### User Interface
- **📱 Mobile Application**: End-user focused app for pitch submissions
- **🌐 Web Dashboard**: HTML/CSS/JavaScript frontend for general users
- **👨‍💻 Developer Dashboard**: Flask-based monitoring and management interface for developers
- **📈 Data Visualization**: Interactive charts and insights dashboard
- **👥 User Management**: Role-based access control and API usage tracking

### DevOps & Monitoring
- **☁️ Cloud Deployment**: Scalable deployment with Docker containerization
- **📊 Weights & Biases**: ML experiment tracking and model monitoring
- **🔍 Real-time Logging**: Comprehensive application and model logging
- **🚦 Health Monitoring**: API health checks and uptime monitoring
- **🔄 Load Testing**: Locust-based performance testing framework

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance async API framework
- **Flask**: Developer dashboard and monitoring interface
- **PyTorch**: Deep learning framework for model inference
- **Transformers**: Hugging Face transformers for NLP models
- **OpenAI Whisper**: Speech-to-text transcription

### Frontend
- **HTML5/CSS3/JavaScript**: Web dashboard frontend
- **Mobile App**: Cross-platform mobile application for end users
- **Flask Templates**: Server-side rendering for developer dashboard

### Database & Storage
- **PostgreSQL**: Primary database (Aiven Cloud)
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migration management

### ML/AI Stack
- **DistilBERT**: Lightweight transformer model (91% accuracy)
- **RoBERTa**: Robust transformer variant (91% accuracy)
- **ALBERT**: Parameter-efficient transformer (91% accuracy)
- **TensorFlow Model**: Custom neural network (90.6% accuracy)
- **Weights & Biases**: Experiment tracking and model monitoring

### Deployment & Monitoring
- **Docker**: Containerized deployment with Dockerfile
- **Locust**: Load testing and performance evaluation
- **Weights & Biases**: ML experiment tracking
- **CORS**: Cross-origin resource sharing

---

## 📁 Project Structure

```
uruti_MLOP/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── Dockerfile                         # Docker container configuration
├── app.py                            # Flask developer dashboard
├── main.py                           # FastAPI ML server
├── models.py                         # Database models
├── locustfile.py                     # Load testing configuration
│
├── URUTI_MODEL-CREATION_Notebook.ipynb # Model training notebook
├── uruti_transcripts.csv             # Training dataset
├── recorded_audio.wav                # Sample audio file
│
├── models/                           # Trained model artifacts
│   ├── distilbert-base-uncased/     # DistilBERT model files
│   ├── roberta-base/                # RoBERTa model files
│   ├── albert-base-v2/              # ALBERT model files
│   └── tensorflow-model/            # TensorFlow model files
│
├── logs_distilbert-base-uncased/     # Training logs for DistilBERT
├── logs_roberta-base/               # Training logs for RoBERTa
├── logs_albert-base-v2/             # Training logs for ALBERT
│
├── results_distilbert-base-uncased/  # Model evaluation results
├── results_roberta-base/           # Model evaluation results
├── results_albert-base-v2/         # Model evaluation results
│
├── wandb/                          # Weights & Biases experiment logs
│
├── templates/                      # Flask HTML templates
│   └── ml_dashboard.html          # Developer dashboard template
│
├── static/                         # Static web assets
│   ├── css/                       # Stylesheets
│   ├── js/                        # JavaScript files
│   └── images/                    # Image assets
│
├── uruti_app/                     # Mobile application code
│
├── uruti-web-app/                 # Web frontend application
│
├── migrations/                    # Database migration files
│
└── __pycache__/                   # Python cache files
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Docker (optional)
- Git
- Node.js (for frontend development)

### 1. Clone the Repository
```bash
git clone https://github.com/NiyonshutiDavid/uruti_MLOP.git
cd uruti_MLOP
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/uruti_db
SECRET_KEY=your_secret_key_here
DEBUG=True
WANDB_API_KEY=your_wandb_api_key
MODEL_PATH=./models/
WHISPER_MODEL=base
```

### 5. Database Setup
```bash
# Initialize database tables
python -c "
import asyncio
from main import init_database
asyncio.run(init_database())
"

# Run migrations
flask db upgrade
```

### 6. Download Pre-trained Models
```bash
# Models will be automatically loaded from the ./models/ directory
# Ensure all model directories are present:
# - distilbert-base-uncased/
# - roberta-base/
# - albert-base-v2/
# - tensorflow-model/
```

---

## 🚀 Running the Application

### Development Mode

#### Start FastAPI ML Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Flask Developer Dashboard
```bash
python3 app.py
```


The services will be available at:
- **ML API**: http://localhost:8000
- **Developer Dashboard**: http://localhost:5000
- **Web Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

### Production Deployment

#### Using Docker
```bash
# Build the image
docker build -t uruti-app .

# Run the container
docker run -p 8000:8000 -p 5000:5000 uruti-app
```

---

## 📱 Demo Screenshots

### Mobile Application (End Users)

| Screenshot | Description |
|------------|-------------|
| <img width="300" alt="Mobile app home screen" src="https://github.com/user-attachments/assets/f2055fbf-b503-4977-89ae-2d91f1080ca6"> | **Home Screen**: Audio recording interface for startup pitch submission |
| <img width="300" alt="Text input screen" src="https://github.com/user-attachments/assets/77cdb2c4-0bd6-414a-8fff-0c48b62d6006"> | **Text Input**: Screen where users can type their startup ideas |
| <img width="300" alt="Prediction results" src="https://github.com/user-attachments/assets/b0e04fd2-54ef-4d77-9126-08909b75bba2"> | **Results**: Real-time prediction showing classification and confidence scores |
| <img width="300" alt="User history" src="https://github.com/user-attachments/assets/9e032869-a472-4bcb-908d-741d82508f6d"> | **History**: Past pitch submissions and feedback |

### Web Application (Admin Dashboard)

| Screenshot | Description |
|------------|-------------|
| <img width="600" alt="Main dashboard" src="https://github.com/user-attachments/assets/ce6e46c9-6381-42ea-a1e7-6d48c70e475a"> | **Main Dashboard**: System metrics and model performance overview |
| <img width="600" alt="Prediction results" src="https://github.com/user-attachments/assets/89148ebb-a7b3-4efb-bafb-f830380eb64d"> | **Results Display**: Real-time prediction results with category breakdown |
| <img width="600" alt="User analytics" src="https://github.com/user-attachments/assets/3e828215-ddba-4d1d-8f4d-e583058e7fe6"> | **Analytics**: User submission history and metrics dashboard |
| <img width="600" alt="Model metrics" src="https://github.com/user-attachments/assets/becc9903-9e9e-4d1f-887f-38fbbf9c4e1d"> | **Model Metrics**: Detailed performance statistics and evaluation |

### Developer Interface

| Screenshot | Description |
|------------|-------------|
| <img width="600" alt="User management" src="https://github.com/user-attachments/assets/b9cd5f3e-dfa7-4fa5-b89f-1f41e8b8aaf3"> | **User Management**: API usage tracking and access control |
| <img width="600" alt="Model training" src="https://github.com/user-attachments/assets/148209c3-6601-4493-813f-33d8c721a665"> | **Model Training**: Retraining interface with Weights & Biases integration |
| <img width="600" alt="System monitoring" src="https://github.com/user-attachments/assets/3200973b-fcf2-43f1-9514-75519c2b8ae4"> | **Monitoring**: Real-time system health indicators |
| <img width="600" alt="Database management" src="https://github.com/user-attachments/assets/c7269d56-ec34-48e8-bc6c-23b2c8d7aed7"> | **Database**: Management and migration interface |
---

## 🔄 ML Pipeline Features

### 1. Data Acquisition & Upload
- **CSV Data Upload**: Support for structured data via `uruti_transcripts.csv`
- **Audio File Processing**: Real-time audio transcription and storage
- **Real-time Data Ingestion**: API endpoints for continuous data collection
- **Data Validation**: Automated quality checks and format validation
- **Storage Management**: Organized data storage with model-specific logging

### 2. Data Preprocessing
- **Text Cleaning**: Advanced NLP preprocessing for startup pitch text
- **Audio Processing**: Whisper-based transcription with noise reduction
- **Feature Engineering**: Token-level feature extraction for transformer models
- **Data Augmentation**: Synthetic data generation for model robustness

### 3. Model Training & Evaluation
- **Multi-Model Ensemble**: Training of 4 different model architectures
- **Transfer Learning**: Fine-tuning pre-trained transformer models
- **Weights & Biases Integration**: Comprehensive experiment tracking
- **Cross-validation**: Robust evaluation across model variants
- **Model Versioning**: Systematic tracking of model iterations

### 4. Model Deployment & Serving
- **Real-time Inference**: Low-latency prediction API with model ensemble
- **Model Selection**: Dynamic selection based on performance metrics
- **Batch Processing**: Bulk prediction capabilities
- **A/B Testing**: Comparison across different model architectures

### 5. Monitoring & Maintenance
- **Weights & Biases Monitoring**: Real-time performance tracking
- **Usage Analytics**: API call patterns and user behavior analysis
- **Model Drift Detection**: Performance degradation alerts
- **Automated Retraining**: Triggered retraining based on performance metrics

---

## 📊 Model Performance & Evaluation

### Current Model Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **DistilBERT-base-uncased** | **91.0%** | **82.0%** | **91.0%** | **86.0%** |
| **RoBERTa-base** | **91.0%** | **82.0%** | **91.0%** | **86.0%** |
| **ALBERT-base-v2** | **91.0%** | **82.0%** | **91.0%** | **86.0%** |
| **TensorFlow Model** | **90.6%** | **82.0%** | **91.0%** | **86.0%** |

### Evaluation Methodology
The model evaluation process includes:

1. **Multi-Model Training**: Four different architectures trained on the same dataset
2. **Comprehensive Metrics**: Accuracy, Precision, Recall, and F1-Score for each model
3. **Weights & Biases Tracking**: All experiments logged and tracked
4. **Cross-Validation**: Consistent evaluation methodology across models
5. **Performance Comparison**: Side-by-side analysis of model architectures

### Model Selection Strategy
- **Ensemble Approach**: Combining predictions from top-performing models
- **Latency Optimization**: DistilBERT selected for production due to speed
- **Accuracy Consistency**: All transformer models achieving 91% accuracy
- **Resource Efficiency**: ALBERT provides parameter efficiency

### Weights & Biases Integration
- **Experiment Tracking**: All training runs logged with hyperparameters
- **Model Comparison**: Visual comparison of model performance
- **Loss Visualization**: Training and validation loss curves
- **Hyperparameter Optimization**: Systematic parameter tuning

---

## 🔄 Retraining Pipeline

### Automated Retraining Triggers
1. **Performance Degradation**: Accuracy drops below 89%
2. **Data Drift Detection**: Statistical changes in input distribution
3. **Schedule-based**: Weekly retraining cycles
4. **Manual Trigger**: On-demand retraining via developer dashboard

### Retraining Process
1. **Data Collection**: Aggregate new labeled data from user feedback
2. **Data Preprocessing**: Clean and prepare new training data
3. **Multi-Model Training**: Retrain all four model architectures
4. **Validation**: Evaluate models against holdout test set
5. **Model Selection**: Choose best performing model for deployment
6. **Weights & Biases Logging**: Track all retraining experiments

### Retraining Monitoring
- **Training Progress**: Real-time training metrics via Weights & Biases
- **Resource Usage**: GPU/CPU utilization during training
- **Model Comparison**: Performance analysis across architectures
- **Experiment History**: Complete training run history and artifacts

---

## 🔍 API Documentation

### Authentication
```bash
# Most endpoints support optional user identification
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your startup pitch here", "user_id": "user@example.com"}'
```

### Core Endpoints

#### Text Prediction
```bash
POST /predict
{
  "text": "We are building an AI-powered solution for small businesses in Rwanda",
  "user_id": "optional_user_id"
}
```

#### Audio Prediction
```bash
POST /predict
Content-Type: multipart/form-data
file: [audio_file.wav]
user_id: optional_user_id
```

#### Model Metrics
```bash
GET /metrics
# Returns performance metrics for all four models
```

#### Training Status
```bash
GET /training/status
POST /training/start
{
  "model_type": "distilbert",  # or "roberta", "albert", "tensorflow"
  "batch_size": 32,
  "learning_rate": 0.001,
  "epochs": 10
}
```

---

## 🧪 Testing & Load Performance

### Load Testing with Locust

#### Running Load Tests
```bash
# Start the locust web interface
locust -f locustfile.py --host=http://localhost:8000

# Or run headless mode
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 300s --headless
```

#### Test Configuration
- **Target URL**: http://localhost:8000
- **Test Duration**: 10 minutes
- **Concurrent Users**: 1, 10, 50, 100, 500
- **Request Types**: Text and Audio predictions

#### Performance Results

##### Single Container Performance
<img width="1440" height="498" alt="Screenshot 2025-08-03 at 18 17 59" src="https://github.com/user-attachments/assets/e22b8965-2b81-4d07-bfdf-a55339b7ad6c" />


##### Model-Specific Performance
- **DistilBERT**: Fastest inference (35ms avg)
- **RoBERTa**: Moderate speed (55ms avg)
- **ALBERT**: Parameter efficient (45ms avg)
- **TensorFlow**: Custom model (40ms avg)

---

## 📈 Data Insights & Visualizations

### Dataset Analysis (`uruti_transcripts.csv`)

#### Feature Distribution Analysis
1. **Text Length Distribution**: Startup pitches range from 20-500 words
2. **Category Balance**: 
   - Mentorship Needed: 35%
   - Investment Ready: 30%
   - Needs Refinement: 35%
3. **Audio Quality Patterns**: Higher quality audio correlates with better transcription

#### Model Performance Insights
1. **Transformer Consistency**: All transformer models achieve identical 91% accuracy
2. **TensorFlow Performance**: Custom model slightly lower at 90.6%
3. **Training Convergence**: Models converge within 5-10 epochs
4. **Feature Importance**: Business model clarity most predictive

#### Weights & Biases Insights
1. **Learning Curves**: Consistent learning patterns across models
2. **Hyperparameter Impact**: Learning rate most critical parameter
3. **Overfitting Detection**: Early stopping prevents overfitting
4. **Resource Usage**: DistilBERT most efficient for production

---

## 🔐 Security & Privacy

### Data Protection
- **Encryption**: All data encrypted in transit and at rest
- **Access Control**: Role-based permissions (End Users vs Developers)
- **Privacy**: User data anonymization and GDPR compliance
- **Audit Logging**: Comprehensive access and modification logs

### Security Measures
- **Input Validation**: Strict validation of all user inputs
- **Rate Limiting**: API call limits to prevent abuse
- **SQL Injection Prevention**: Parameterized queries and ORM usage
- **CORS Configuration**: Restricted cross-origin requests

---
## Disclaimer
Notebook can't be viewed on github, but you can check it directly from google colab for it's copy that is on drive:
[https://colab.research.google.com/drive/1YMs-V08EZYY6n_okgVKjuiJXstFkmQSL?usp=sharing]

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- **Python Style**: Follow PEP 8 guidelines
- **Documentation**: Comprehensive docstrings for all functions
- **Testing**: Write tests for new features and bug fixes
- **Weights & Biases**: Log all experiments and model changes

---

## 📞 Support & Contact

### Project Team
- **Lead Developer**: David Niyonshuti (d.niyonshut@alustudent.com)
- **ML Engineer**: David Niyonshuti (d.niyonshut@alustudent.com)
- **Frontend Developer**: David Niyonshuti (d.niyonshut@alustudent.com)

### Getting Help
- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Check our comprehensive wiki
- **Community**: Join our Slack workspace
- **Email**: Contact support@uruti.rw

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face**: For the incredible Transformers library and pre-trained models
- **OpenAI**: For the Whisper speech recognition model
- **Weights & Biases**: For comprehensive ML experiment tracking
- **FastAPI Team**: For the high-performance web framework
- **Aiven**: For reliable cloud database services
- **Wandb.ai**: For model performance over api

---

## 🎬 Demo Video

**YouTube Demo**: [[YOUTUBE AVAILABLE LINK HERE](https://youtu.be/eFr4CMAEWBs)]

The demo video showcases:
- Complete end-to-end user journey on mobile app
- Web dashboard functionality for general users
- Developer dashboard with model monitoring
- Real-time prediction and audio processing
- Multi-model comparison and performance metrics
- Model retraining with Weights & Biases integration
- Load testing results and scalability demonstration

---

*Last updated: 03-08-2025*
*Version: 2.1.0*
*Models: DistilBERT, RoBERTa, ALBERT, TensorFlow (91% accuracy)*
