# Makefile for Retail Churn & Forecast Project

install:
	pip install -r requirements.txt

data:
	# Command to download data from Kaggle if not present
	python -c "import kagglehub; kagglehub.dataset_download('mashlyn/online-retail-ii-uci')" 

train:
	python main.py --data data/online_retail_II.csv

clean:
	rm -rf models/*.pkl
	rm -rf data/processed/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} +

# The "One-Click" command for recruiters
all: install train


serve:
	python app.py


# Existing commands...

docker-build:
	docker build -t retail-churn-api .

docker-run:
	docker run -p 8000:8000 retail-churn-api

# Add this to your existing Makefile
up:
	docker-compose up --build

down:
	docker-compose down