FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (caching)
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose Flask port
EXPOSE 8000

# Set environment variable for Flask to run on jup 
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000 

# Start Flask
CMD ["flask", "run"]
