.PHONY: setup run test clean

setup:
	conda env create -f environment.yml
	python -m ipykernel install --user --name rag-eval --display-name "Python (rag-eval)"

run:
	docker-compose up

test:
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete