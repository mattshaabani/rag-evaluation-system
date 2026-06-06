# docker/Dockerfile
# Production image for the RAG API

# Use official Python slim image — smaller than full Python
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
# gcc needed for some Python packages that compile C extensions
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first
# Docker caches layers — if requirements don't change,
# pip install is skipped on rebuild (much faster)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .

# Install the project as a package
RUN pip install -e .

# Expose the API port
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]