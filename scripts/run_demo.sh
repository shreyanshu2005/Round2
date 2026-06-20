#!/usr/bin/env bash
# scripts/run_demo.sh
# One-click BTIP demo startup: infra -> seed -> train -> backend -> frontend
set -e

echo "=== BTIP One-Click Demo ==="

echo "[1/5] Starting Docker infra (postgres, redis, mlflow)..."
docker-compose up -d postgres redis mlflow

echo "[2/5] Waiting for Postgres to be healthy..."
until docker-compose exec -T postgres pg_isready -U btip > /dev/null 2>&1; do
  sleep 1
done

echo "[3/5] Seeding database..."
python scripts/seed_db.py

echo "[4/5] Training all models (this may take a while)..."
python scripts/train_all_models.py

echo "[5/5] Launching backend + frontend..."
(cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload &)
sleep 3
(cd frontend && npm run dev &)

echo ""
echo "BTIP is up:"
echo "  Backend API : http://localhost:8000/docs"
echo "  GraphQL     : http://localhost:8000/graphql"
echo "  Frontend    : http://localhost:3000"
echo "  MLflow      : http://localhost:5000"