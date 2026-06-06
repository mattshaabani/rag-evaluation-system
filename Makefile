.PHONY: setup run stop test clean mlflow api

setup:
	conda env create -f environment.yml
	python -m ipykernel install --user --name rag-eval --display-name "Python (rag-eval)"
	pip install -e .

# Run everything with Docker
run:
	docker-compose up --build

# Run in background
run-detached:
	docker-compose up --build -d

# Stop everything
stop:
	docker-compose down

# Run API locally without Docker
api:
	uvicorn src.api.main:app --reload --port 8080

# Open MLflow UI locally
mlflow:
	mlflow ui --port 5000

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	docker-compose down -v