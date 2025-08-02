from fastapi import FastAPI, File, UploadFile
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Load Whisper ASR model
whisper_model = whisper.load_model("base")

# Load classification model and tokenizer
model_path = './models/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Uruti ML API"}

@app.post("/predict")
async def predict(text: str | None = None, file: UploadFile | None = None):
    if not (text or file) or (text and file):
        raise HTTPException(status_code=400, detail="Provide either text or a file, but not both.")

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Whisper ASR model
whisper_model = whisper.load_model("base")

# Load classification model and tokenizer
model_path = './models/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define labels
labels = ['Mentorship Needed', 'Investment Ready', 'Needs Refinement']

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Uruti ML API"}

@app.post("/predict")
async def predict(text: str | None = None, file: UploadFile | None = None):
    if not (text or file) or (text and file):
        raise HTTPException(status_code=400, detail="Provide either text or a file, but not both.")

    input_text = ""
    if file:
        # Handle audio file upload
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(await file.read())
                tmp_file_path = tmp_file.name

            # Transcribe the audio
            result = whisper_model.transcribe(tmp_file_path)
            input_text = result["text"]

            # Clean up the temporary file
            os.remove(tmp_file_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio file: {e}")
    elif text:
        # Handle text input
        input_text = text

    if not input_text.strip():
         # Handle empty or whitespace-only input after transcription or from text input
        return {"transcribed_text": input_text, "predicted_label": "Needs Refinement", "message": "Input was empty, defaulting to 'Needs Refinement'."}


    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Map the predicted class ID back to the label
    predicted_label = labels[predicted_class_id]

    return {"transcribed_text": input_text if file else None, "predicted_label": predicted_label}
