FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install jupyter

# Copy the rest of your app
COPY . .

# Expose the default Jupyter port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
