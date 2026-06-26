FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Create weights directory
RUN mkdir -p weights

# Expose ports
EXPOSE 8000 8501

# Start script — runs FastAPI + Streamlit together
COPY app/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
