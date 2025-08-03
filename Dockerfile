# Use Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system packages (for psycopg2 and others)
RUN apt-get update && apt-get install -y \
    build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose both FastAPI and Flask ports
EXPOSE 8000
EXPOSE 5000

# Run both apps in background using a script
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & python3 uruti-web-app/app.py"]
