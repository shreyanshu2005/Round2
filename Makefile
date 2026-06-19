.PHONY: up down seed train test lint demo logs

up:
	docker-compose up -d

down:
	docker-compose down

seed:
	python scripts/seed_db.py

train:
	python scripts/train_all_models.py

features:
	python scripts/build_feature_store.py

test:
	pytest backend/tests/ -v

lint:
	ruff backend/
	cd frontend && npx eslint src/

logs:
	docker-compose logs -f

demo:
	bash scripts/run_demo.sh
